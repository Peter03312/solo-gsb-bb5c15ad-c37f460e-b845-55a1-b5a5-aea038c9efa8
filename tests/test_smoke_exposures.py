"""烟雾暴露时段与人群组累计暴露上限集成测试。

覆盖：
* 按通道登记左闭右开烟雾时段（值非负、从当前分钟起、同通道不重叠），
  未登记分钟暴露为零、节点等待不计入；
* 为未出发组登记非负累计暴露上限（未设上限的组不受约束、再登记替换）；
* 编排在暴露不超限的路线中沿用封闭/容量/截止/先导放行与择优规则，
  可等待烟雾消散后通行；上限内无可行路线归因 BLOCKED_BY_SMOKE；
* 登记使已出发组按固定路线预计累计暴露超限时 409 整次拒绝
  （不改编排、不登记时段、不产生修订）；
* 节点滞留改编保留已走前缀按原计划时刻计算的暴露、只用剩余额度寻找
  余程，无可行余程沿用待救援流程；
* /plan 展示各组上限、前缀已用与余程预计暴露；
* 成功登记进入既有修订（参数、前后差异、替换记录），场景重置清空。
"""
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

_tmpdir = tempfile.mkdtemp(prefix="evac-test-smoke-")
os.environ["EVAC_DB_PATH"] = os.path.join(_tmpdir, "test.db")
os.environ["EVAC_SEED"] = "false"

from app.main import app  # noqa: E402


SCENARIO = {
    "passages": [
        {"id": 1, "src": 1, "dst": 2, "minutes": 4, "capacity": 30},
        {"id": 2, "src": 2, "dst": 3, "minutes": 3, "capacity": 30},
        {"id": 3, "src": 3, "dst": 5, "minutes": 2, "capacity": 30},
        {"id": 5, "src": 2, "dst": 6, "minutes": 2, "capacity": 40},
    ],
    "exits": [5, 6],
    "closures": [],
    "groups": [
        {"id": 1, "size": 10, "origin": 1, "earliest_departure": 0,
         "priority": 10, "deadline": 40},
        {"id": 3, "size": 10, "origin": 2, "earliest_departure": 0,
         "priority": 20, "deadline": 20},
        {"id": 4, "size": 10, "origin": 1, "earliest_departure": 0,
         "priority": 5, "deadline": 40},
    ],
}


@pytest.fixture()
def client():
    with TestClient(app) as c:
        resp = c.post("/scenario", json=SCENARIO)
        assert resp.status_code == 200
        yield c


def by_id(plan):
    return {g["group_id"]: g for g in plan["groups"]}


def windows(c):
    return c.get("/smoke-exposures").json()["windows"]


def limits(c):
    return c.get("/group-smoke-limits").json()["limits"]


def revs(c):
    return c.get("/revisions").json()["revisions"]


def set_limit(c, gid, value):
    return c.post("/group-smoke-limits",
                  json={"group_id": gid, "smoke_limit": value})


def add_window(c, edge_id, a, b, value):
    return c.post("/smoke-exposures", json={
        "edge_id": edge_id, "start_minute": a, "end_minute": b,
        "exposure": value})


# --------------------------------------------------------------------------
# 未设上限：登记时段不改变编排，仅在 /plan 中不展示暴露字段
# --------------------------------------------------------------------------

def test_windows_without_limits_do_not_constrain(client):
    resp = add_window(client, 1, 0, 40, 100)
    assert resp.status_code == 201
    groups = by_id(resp.json())
    # 组 1 原走通道 1、5：通道 1[0,4) 暴露 400 但未设上限，照常 t=0 出发
    assert groups[1]["route"] == [1, 5]
    assert groups[1]["departure_minute"] == 0
    assert groups[1]["smoke_limit"] is None
    assert groups[1]["smoke_prefix_exposure"] is None
    assert groups[1]["smoke_remaining_exposure"] is None
    assert windows(client) == [{
        "edge_id": 1, "start_minute": 0, "end_minute": 40, "exposure": 100}]


def test_zero_exposure_window_is_neutral(client):
    # 值为零的时段登记成功、可与其他窗口并列，编排不受影响
    assert add_window(client, 5, 0, 10, 0).status_code == 201
    groups = by_id(client.get("/plan").json())
    assert groups[3]["route"] == [5]
    assert groups[3]["departure_minute"] == 0
    set_limit(client, 3, 0)
    groups = by_id(client.get("/plan").json())
    assert groups[3]["status"] == "scheduled"
    assert groups[3]["smoke_prefix_exposure"] == 0
    assert groups[3]["smoke_remaining_exposure"] == 0


# --------------------------------------------------------------------------
# 上限约束：等待烟雾消散 / 绕行 / 无可行路线
# --------------------------------------------------------------------------

def test_group_waits_for_smoke_to_dissipate(client):
    # 组 3（节点2，原走通道5[0,2) 到出口6）上限 3；
    # 通道5 [0,2) 每分钟2 => 占用即累计4 > 3，等到第2分钟（窗口外）进入
    set_limit(client, 3, 3)
    resp = add_window(client, 5, 0, 2, 2)
    assert resp.status_code == 201
    g = by_id(resp.json())[3]
    assert g["status"] == "scheduled"
    assert g["route"] == [5]
    assert g["departure_minute"] == 2
    assert g["arrival_minute"] == 4
    # 节点等待（0..2）不计暴露；窗口外通道占用暴露为 0
    assert g["smoke_prefix_exposure"] == 0
    assert g["smoke_remaining_exposure"] == 0


def test_group_takes_detour_when_waiting_is_slower(client):
    # 通道5 [0,10) 每分钟2、上限3：等到10到达12；改道通道2、3：
    # 通道2[0,3) 暴露6 超上限 -> 等到3进入[3,6)（窗口外，0），再通道3，
    # 到达8，更早
    set_limit(client, 3, 3)
    resp = add_window(client, 5, 0, 10, 2)
    assert resp.status_code == 201
    resp = add_window(client, 2, 0, 3, 2)
    assert resp.status_code == 201
    g = by_id(resp.json())[3]
    assert g["status"] == "scheduled"
    assert g["route"] == [2, 3]
    assert [s["enter_minute"] for s in g["steps"]] == [3, 6]
    assert g["arrival_minute"] == 8
    assert g["smoke_remaining_exposure"] == 0


def test_partial_budget_spent_on_first_edge_then_wait(client):
    # 组 1 走通道1（4分钟）接通道5（2分钟），上限5：
    # 通道1 [0,4) 每分钟1（占用累计4，余1）；通道5 [4,6) 每分钟2（再累计
    # 2 => 总6 > 5），通道5 等到6进入[6,8)（窗口外0），到达8
    set_limit(client, 1, 5)
    assert add_window(client, 1, 0, 4, 1).status_code == 201
    resp = add_window(client, 5, 4, 6, 2)
    assert resp.status_code == 201
    g = by_id(resp.json())[1]
    assert g["route"] == [1, 5]
    assert [s["enter_minute"] for s in g["steps"]] == [0, 6]
    assert g["arrival_minute"] == 8
    assert g["smoke_remaining_exposure"] == 4


def test_blocked_by_smoke_when_no_route_within_budget(client):
    # 组 1 的唯一首条通道 1：[0,40) 每分钟1，通行4分钟即累计4；
    # 上限3 => 截止前任何时刻进入都超限
    set_limit(client, 1, 3)
    resp = add_window(client, 1, 0, 40, 1)
    assert resp.status_code == 201
    g = by_id(resp.json())[1]
    assert g["status"] == "failed"
    assert g["reason"] == "BLOCKED_BY_SMOKE"
    assert g["smoke_prefix_exposure"] == 0
    assert g["smoke_remaining_exposure"] is None


def test_smoke_combined_with_closure_keeps_earliest_arrival(client):
    # 封闭通道5 [0,8)：组3 若走通道5 最早8进入、10到达；改道通道2、3：
    # 通道2 [0,3) 每分钟1（上限2 不可进入），等到3进入[3,6)（窗口外0），
    # 再通道3，到达8，更早
    set_limit(client, 3, 2)
    assert client.post("/closures", json={
        "edge_id": 5, "start_minute": 0, "end_minute": 8}).status_code == 201
    resp = add_window(client, 2, 0, 3, 1)
    assert resp.status_code == 201
    g = by_id(resp.json())[3]
    assert g["route"] == [2, 3]
    assert [s["enter_minute"] for s in g["steps"]] == [3, 6]
    assert g["arrival_minute"] == 8


def test_limit_replace_replans(client):
    set_limit(client, 1, 0)
    assert add_window(client, 1, 0, 10, 1).status_code == 201
    g = by_id(client.get("/plan").json())[1]
    assert g["status"] == "scheduled"
    assert g["departure_minute"] == 10
    # 放宽上限为 4：第 0 分钟进入通道1[0,4) 累计4 可行
    resp = set_limit(client, 1, 4)
    assert resp.status_code == 201
    g = by_id(resp.json())[1]
    assert g["departure_minute"] == 0
    assert g["smoke_limit"] == 4
    assert limits(client) == [{"group_id": 1, "smoke_limit": 4}]


# --------------------------------------------------------------------------
# 登记校验：422 / 409
# --------------------------------------------------------------------------

def test_window_validation_errors(client):
    assert add_window(client, 999, 0, 2, 1).status_code == 422
    assert add_window(client, 1, 2, 2, 1).status_code == 422
    assert add_window(client, 1, 3, 2, 1).status_code == 422
    assert add_window(client, 1, 0, 2, -1).status_code == 422
    assert windows(client) == []


def test_window_start_before_current_minute_rejected(client):
    assert client.post("/tick", json={"minutes": 3}).status_code == 200
    resp = add_window(client, 1, 2, 5, 1)
    assert resp.status_code == 422
    # 起点等于当前分钟允许
    assert add_window(client, 1, 3, 5, 1).status_code == 201


def test_window_overlap_rejected(client):
    assert add_window(client, 5, 5, 10, 2).status_code == 201
    for a, b in ((8, 12), (0, 6), (6, 8), (0, 30)):
        resp = add_window(client, 5, a, b, 2)
        assert resp.status_code == 409
        body = resp.json()
        assert body["error"] == "SMOKE_EXPOSURE_OVERLAP"
        assert body["conflicts"][0]["existing"] == [5, 10]
        assert body["conflicts"][0]["requested"] == [a, b]
    # 左闭右开相接不重叠
    assert add_window(client, 5, 10, 12, 2).status_code == 201
    # 不同通道互不相干
    assert add_window(client, 2, 5, 10, 2).status_code == 201


def test_limit_unknown_or_negative_rejected(client):
    assert set_limit(client, 999, 1).status_code == 422
    resp = client.post("/group-smoke-limits",
                       json={"group_id": 1, "smoke_limit": -1})
    assert resp.status_code == 422
    assert limits(client) == []


# --------------------------------------------------------------------------
# 已出发组：登记使固定路线预计累计暴露超限 -> 409 整次拒绝
# --------------------------------------------------------------------------

def test_window_overloading_departed_group_rejected(client):
    # 组 3（10人）t=0 进入通道5，占用 [0,2)
    set_limit(client, 3, 3)
    assert client.post("/tick", json={"minutes": 1}).status_code == 200
    plan_before = client.get("/plan").json()
    n0 = len(revs(client))

    # 通道5 [1,2) 每分钟4：固定路线 [0,2) 预计暴露 4 > 3 -> 409
    resp = add_window(client, 5, 1, 2, 4)
    assert resp.status_code == 409
    body = resp.json()
    assert body["error"] == "SMOKE_EXPOSURE_OVERLOADS_DEPARTED_GROUP"
    conflict = body["conflicts"][0]
    assert conflict["group_id"] == 3
    assert conflict["smoke_limit"] == 3
    assert conflict["projected_exposure"] == 4
    assert conflict["violating_edge_id"] == 5
    assert conflict["window"] == [1, 2]
    assert conflict["violating_minutes"] == [1]

    # 整次拒绝：不登记、不改编排、不产生修订
    assert windows(client) == []
    assert client.get("/plan").json() == plan_before
    assert len(revs(client)) == n0

    # 固定路线能容纳（每分钟3，全程3 <= 上限3）时允许登记
    assert add_window(client, 5, 1, 2, 3).status_code == 201
    g = by_id(client.get("/plan").json())[3]
    assert g["status"] == "fixed" and g["route"] == [5]
    assert g["smoke_prefix_exposure"] == 0  # 前缀 [0,1) 未登记=0
    assert g["smoke_remaining_exposure"] == 3


def test_window_missing_fixed_route_is_allowed(client):
    set_limit(client, 1, 0)
    # 组 1 走通道1、5；通道2 不在其固定路线上，任意值都不冲突
    assert client.post("/tick", json={"minutes": 1}).status_code == 200
    assert add_window(client, 2, 1, 20, 99).status_code == 201
    assert add_window(client, 3, 1, 20, 99).status_code == 201


def test_departed_group_limit_registration_rejected(client):
    assert client.post("/tick", json={"minutes": 1}).status_code == 200
    resp = set_limit(client, 3, 1)
    assert resp.status_code == 409
    body = resp.json()
    assert body["error"] == "GROUP_ALREADY_DEPARTED"
    assert body["conflicts"][0]["group_id"] == 3
    assert body["conflicts"][0]["reason"] == "TARGET_ALREADY_DEPARTED"
    assert limits(client) == []


def test_full_route_includes_prefix_in_departed_check(client):
    # 固定路线的检查含已走前缀（按原计划时刻）：组1 通道1[0,4)+通道5[4,6)，
    # 上限5；第1分钟后登记通道1 [1,4) 每分钟1（剩余前缀累计3，不冲突），
    # 再登记通道5 [4,5) 每分钟1（全程4 <= 5，允许）、[5,6) 每分钟2：
    # 全程 3+1+2=6 > 5 -> 冲突
    set_limit(client, 1, 5)
    assert client.post("/tick", json={"minutes": 1}).status_code == 200
    assert add_window(client, 1, 1, 4, 1).status_code == 201
    assert add_window(client, 5, 4, 5, 1).status_code == 201
    resp = add_window(client, 5, 5, 6, 2)
    assert resp.status_code == 409
    conflict = resp.json()["conflicts"][0]
    assert conflict["projected_exposure"] == 6
    assert conflict["violating_minutes"] == [5]
    assert windows(client) == [
        {"edge_id": 1, "start_minute": 1, "end_minute": 4, "exposure": 1},
        {"edge_id": 5, "start_minute": 4, "end_minute": 5, "exposure": 1}]


# --------------------------------------------------------------------------
# /plan 展示：固定组前缀已用与余程预计
# --------------------------------------------------------------------------

def test_plan_shows_prefix_and_remaining_after_tick(client):
    set_limit(client, 1, 20)
    # 通道1 [0,4) 每分钟2（前缀全程8）；通道5 [4,6) 每分钟1：
    # 截止前最早到达规则下组1 仍第4分钟进入（余程暴露2，早到优于等待），
    # 全程10 <= 上限20
    add_window(client, 1, 0, 4, 2)
    add_window(client, 5, 4, 6, 1)
    # 推进到第5分钟：组1 fixed，通道1 已走完（前缀暴露8），通道5 尚在
    # 剩余路线（计划 [4,6)，当前正在通行），余程预计暴露2
    client.post("/tick", json={"minutes": 5})
    g = by_id(client.get("/plan").json())[1]
    assert g["status"] == "fixed"
    assert g["smoke_prefix_exposure"] == 8
    assert g["smoke_remaining_exposure"] == 2
    # 推进到抵达出口后：前缀为全程暴露10，余程为 0
    client.post("/tick", json={"minutes": 1})
    g = by_id(client.get("/plan").json())[1]
    assert g["smoke_prefix_exposure"] == 10
    assert g["smoke_remaining_exposure"] == 0


# --------------------------------------------------------------------------
# 节点滞留改编：前缀额度 + 余程
# --------------------------------------------------------------------------

DELAY_SCENARIO = {
    "passages": [
        {"id": 1, "src": 1, "dst": 2, "minutes": 4, "capacity": 10},
        {"id": 2, "src": 2, "dst": 6, "minutes": 2, "capacity": 10},
    ],
    "exits": [5, 6],
    "closures": [],
    "groups": [
        {"id": 1, "size": 5, "origin": 1, "earliest_departure": 0,
         "priority": 10, "deadline": 30}],
}


def test_delay_report_keeps_prefix_exposure_and_uses_remaining_budget(client):
    client.post("/scenario", json=DELAY_SCENARIO)
    # 上限5：通道1[0,4) 每分钟1（前缀4，余1）；通道2 无窗口
    set_limit(client, 1, 5)
    add_window(client, 1, 0, 4, 1)
    client.post("/tick", json={"minutes": 4})
    # 通道2 [4,6) 零暴露登记后固定路线仍可行；再登记 [6,20) 每分钟1
    add_window(client, 2, 4, 6, 0)
    add_window(client, 2, 6, 20, 1)
    # 滞留、第4分钟恢复：余程 [4,6) 暴露0 -> 立即通行
    resp = client.post("/groups/1/delay-report",
                       json={"node_id": 2, "recovery_minute": 4})
    g = by_id(resp.json())[1]
    assert g["status"] == "fixed"
    assert g["prefix_step_count"] == 1
    assert g["smoke_prefix_exposure"] == 4
    assert g["smoke_remaining_exposure"] == 0
    assert g["steps"][1]["enter_minute"] == 4

    # 再次滞留、第6分钟恢复：余程额度1，[6,8) 每分钟1 累计2 > 1，
    # 等到20进入[20,22)（窗口外0），到达22 <= 截止30
    resp = client.post("/groups/1/delay-report",
                       json={"node_id": 2, "recovery_minute": 6})
    assert resp.status_code == 200
    g = by_id(resp.json())[1]
    assert g["status"] == "fixed"
    assert g["steps"][1]["enter_minute"] == 20
    assert g["smoke_prefix_exposure"] == 4
    assert g["smoke_remaining_exposure"] == 0


def test_delay_report_no_remaining_route_goes_awaiting_rescue(client):
    client.post("/scenario", json=DELAY_SCENARIO)
    # 上限4：前缀通道1[0,4) 每分钟1 恰用4，余0；通道2 [6,30) 每分钟1
    # （不触及固定路线 [4,6)，可在第4分钟后登记）
    set_limit(client, 1, 4)
    add_window(client, 1, 0, 4, 1)
    client.post("/tick", json={"minutes": 4})
    assert add_window(client, 2, 6, 30, 1).status_code == 201
    # 第6分钟恢复：余程额度0，通道2 [6,8) 每分钟1（累计2 > 0），
    # 等待到窗口结束30再通行则到达32 > 截止30 -> 无可行余程
    resp = client.post("/groups/1/delay-report",
                       json={"node_id": 2, "recovery_minute": 6})
    g = by_id(resp.json())[1]
    assert g["status"] == "awaiting_rescue"
    assert g["reason"] == "BLOCKED_BY_SMOKE"
    assert g["prefix_step_count"] == 1
    assert g["smoke_prefix_exposure"] == 4
    assert g["smoke_remaining_exposure"] is None


def test_delay_report_rescued_after_smoke_blocked(client):
    # 无可行余程沿用待救援流程：可交接人工救援队并结案
    client.post("/scenario", json=DELAY_SCENARIO)
    set_limit(client, 1, 0)
    client.post("/tick", json={"minutes": 4})
    assert add_window(client, 2, 6, 30, 1).status_code == 201
    resp = client.post("/groups/1/delay-report",
                       json={"node_id": 2, "recovery_minute": 6})
    assert resp.json()["groups"][0]["status"] == "awaiting_rescue"
    assert client.post("/rescue-teams",
                       json={"team_id": 1, "capacity": 10}).status_code == 201
    resp = client.post("/rescue-tasks", json={
        "group_id": 1, "team_id": 1, "node_id": 2,
        "observed_minute": 4})
    assert resp.status_code == 201
    g = by_id(client.get("/plan").json())[1]
    assert g["status"] == "rescuing"
    assert g["smoke_prefix_exposure"] == 0
    assert g["smoke_remaining_exposure"] is None


# --------------------------------------------------------------------------
# 修订与重置
# --------------------------------------------------------------------------

def test_smoke_changes_create_revisions(client):
    n0 = len(revs(client))
    set_limit(client, 1, 7)
    add_window(client, 1, 0, 4, 1)
    entries = revs(client)
    assert len(entries) == n0 + 2
    assert entries[-2]["operation"] == "group-smoke-limits"
    assert entries[-2]["detail"] == {"group_id": 1, "smoke_limit": 7}
    assert entries[-1]["operation"] == "smoke-exposures"
    assert entries[-1]["detail"] == {
        "edge_id": 1, "start_minute": 0, "end_minute": 4, "exposure": 1}

    detail = client.get(f"/revisions/{entries[-1]['rev']}").json()
    changes = {ch["group_id"]: ch for ch in detail["changes"]}
    assert 1 in changes
    # 组1 通道1[0,4) 累计4 <= 7，路线不变（无差异时不在 changes 中也允许）


def test_rejected_window_creates_no_revision(client):
    set_limit(client, 3, 0)
    client.post("/tick", json={"minutes": 1})
    n0 = len(revs(client))
    # 使已出发组超载 -> 409，不产生修订
    assert add_window(client, 5, 1, 2, 1).status_code == 409
    assert len(revs(client)) == n0
    # 重叠 -> 409，不产生修订
    assert add_window(client, 1, 1, 3, 0).status_code == 201
    n1 = len(revs(client))
    assert add_window(client, 1, 2, 4, 0).status_code == 409
    assert len(revs(client)) == n1


def test_scenario_reset_clears_smoke_settings(client):
    set_limit(client, 1, 3)
    add_window(client, 1, 0, 5, 1)
    rev = revs(client)[-1]["rev"]
    detail_before = client.get(f"/revisions/{rev}").json()

    assert client.post("/scenario", json=SCENARIO).status_code == 200
    assert windows(client) == []
    assert limits(client) == []
    g = by_id(client.get("/plan").json())[1]
    assert g["smoke_limit"] is None
    assert g["smoke_prefix_exposure"] is None
    assert g["smoke_remaining_exposure"] is None
    # 旧修订保留、内容不变
    assert client.get(f"/revisions/{rev}").json() == detail_before
