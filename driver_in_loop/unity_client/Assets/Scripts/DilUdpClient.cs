using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

[Serializable]
public class DilEgoState
{
    public float x;
    public float y;
    public float yaw;
    public float speed;
    public float acceleration;
    public float steer;
    public float length;
    public float width;
}

[Serializable]
public class DilVehicleState
{
    public string id;
    public int slot;
    public string name;
    public float x;
    public float y;
    public float yaw;
    public float length;
    public float width;
}

[Serializable]
public class DilAuthorityState
{
    public float reference;
    public float rl;
}

[Serializable]
public class DilTrustState
{
    public float human_to_machine;
    public float machine_to_human;
}

[Serializable]
public class DilRiskState
{
    public float environment_urgency;
    public float front_distance_m;
    public float ttc_s;
}

[Serializable]
public class DilSafetyState
{
    public bool collision;
}

[Serializable]
public class DilRoadState
{
    public float[] lane_markings;
    public int ego_lane_id;
    public int driving_direction;
}

[Serializable]
public class DilPointState
{
    public float x;
    public float y;
}

[Serializable]
public class DilIntentionState
{
    public DilPointState[] machine;
    public DilPointState[] human;
}

[Serializable]
public class DilDriverInputState
{
    public string source;
    public float steer;
    public float throttle;
    public float brake;
    public float delta_rad;
    public float acceleration_mps2;
}

[Serializable]
public class DilSimState
{
    public string type;
    public string session_id;
    public string mode;
    public int paper_case_id;
    public int case_id;
    public string case_name;
    public int frame_index;
    public float time_s;
    public float dt;
    public DilEgoState ego;
    public DilVehicleState[] vehicles;
    public DilIntentionState intention;
    public DilAuthorityState authority;
    public DilTrustState trust;
    public DilRiskState risk;
    public DilSafetyState safety;
    public DilRoadState road;
    public DilDriverInputState driver_input;
}

public class DilUdpClient : MonoBehaviour
{
    public int listenPort = 50710;
    public bool useRenderInterpolation = true;
    public float renderDelaySeconds = 0.06f;
    public int maxBufferedStates = 240;
    public DilSimState LatestState { get; private set; }
    public bool HasState { get; private set; }

    private UdpClient udp;
    private Thread receiveThread;
    private volatile bool running;
    private readonly object stateLock = new object();
    private readonly List<DilSimState> stateHistory = new List<DilSimState>();
    private int lastAcceptedFrameIndex = -1;
    private float lastAcceptedTime = -1.0f;
    private int stateSequence;
    private bool renderClockReady;
    private float renderClock;
    private string activeSessionId = "";

    void Start()
    {
        udp = new UdpClient(listenPort);
        running = true;
        receiveThread = new Thread(ReceiveLoop);
        receiveThread.IsBackground = true;
        receiveThread.Start();
        Debug.Log($"DIL UDP client listening on port {listenPort}");
    }

    void ReceiveLoop()
    {
        IPEndPoint remote = new IPEndPoint(IPAddress.Any, 0);
        while (running)
        {
            try
            {
                byte[] data = udp.Receive(ref remote);
                string json = Encoding.UTF8.GetString(data);
                DilSimState parsed = JsonUtility.FromJson<DilSimState>(json);
                if (parsed == null || parsed.ego == null) continue;

                lock (stateLock)
                {
                    string incomingSession = parsed.session_id == null ? "" : parsed.session_id;
                    bool hasIncomingSession = incomingSession.Length > 0;
                    bool hasActiveSession = activeSessionId != null && activeSessionId.Length > 0;
                    bool shouldSwitchSession =
                        hasIncomingSession &&
                        (!hasActiveSession || parsed.time_s < 0.25f && incomingSession != activeSessionId);

                    if (shouldSwitchSession)
                    {
                        activeSessionId = incomingSession;
                        stateHistory.Clear();
                        lastAcceptedFrameIndex = -1;
                        lastAcceptedTime = -1.0f;
                        renderClockReady = false;
                    }
                    else if (hasIncomingSession && hasActiveSession && incomingSession != activeSessionId)
                    {
                        continue;
                    }

                    // UDP may occasionally deliver an older packet after a newer one.
                    // Do not clear the buffer for normal out-of-order packets; just ignore them.
                    bool looksLikeNewRun =
                        lastAcceptedTime >= 0.0f &&
                        parsed.time_s < lastAcceptedTime - 0.5f;

                    if (looksLikeNewRun)
                    {
                        stateHistory.Clear();
                        lastAcceptedFrameIndex = -1;
                        lastAcceptedTime = -1.0f;
                        renderClockReady = false;
                    }
                    else if (
                        lastAcceptedFrameIndex >= 0 &&
                        parsed.frame_index <= lastAcceptedFrameIndex &&
                        parsed.time_s <= lastAcceptedTime
                    )
                    {
                        continue;
                    }

                    LatestState = parsed;
                    HasState = true;
                    lastAcceptedFrameIndex = parsed.frame_index;
                    lastAcceptedTime = parsed.time_s;
                    stateSequence += 1;

                    stateHistory.Add(parsed);
                    while (stateHistory.Count > maxBufferedStates)
                    {
                        stateHistory.RemoveAt(0);
                    }
                }
            }
            catch (SocketException)
            {
                if (running) Debug.LogWarning("DIL UDP receive socket closed.");
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"DIL UDP parse failed: {ex.Message}");
            }
        }
    }

    public bool TryGetState(out DilSimState state)
    {
        lock (stateLock)
        {
            state = LatestState;
            return HasState;
        }
    }

    public bool TryGetRenderState(out DilSimState state)
    {
        lock (stateLock)
        {
            state = LatestState;
            if (!HasState || LatestState == null) return false;
            if (!useRenderInterpolation || stateHistory.Count < 2) return true;

            float latestPlayableTime = LatestState.time_s - Mathf.Max(0.0f, renderDelaySeconds);
            if (!renderClockReady)
            {
                renderClock = Mathf.Max(stateHistory[0].time_s, latestPlayableTime);
                renderClockReady = true;
            }
            else
            {
                renderClock += Time.deltaTime;
                renderClock = Mathf.Min(renderClock, latestPlayableTime);
            }

            float targetTime = renderClock;
            if (targetTime <= stateHistory[0].time_s)
            {
                state = stateHistory[0];
                return true;
            }

            for (int i = stateHistory.Count - 2; i >= 0; --i)
            {
                DilSimState a = stateHistory[i];
                DilSimState b = stateHistory[i + 1];
                if (a.time_s <= targetTime && targetTime <= b.time_s)
                {
                    float denom = Mathf.Max(1e-4f, b.time_s - a.time_s);
                    float alpha = Mathf.Clamp01((targetTime - a.time_s) / denom);
                    state = InterpolateState(a, b, alpha, targetTime);
                    return true;
                }
            }

            return true;
        }
    }

    DilSimState InterpolateState(DilSimState a, DilSimState b, float alpha, float targetTime)
    {
        DilSimState s = new DilSimState();
        s.type = b.type;
        s.mode = b.mode;
        s.paper_case_id = b.paper_case_id;
        s.case_id = b.case_id;
        s.case_name = b.case_name;
        s.frame_index = b.frame_index;
        s.time_s = targetTime;
        s.dt = b.dt;
        s.ego = InterpolateEgo(a.ego, b.ego, alpha);
        s.vehicles = InterpolateVehicles(a.vehicles, b.vehicles, alpha);
        s.intention = b.intention;
        s.authority = new DilAuthorityState
        {
            reference = Mathf.Lerp(a.authority.reference, b.authority.reference, alpha),
            rl = Mathf.Lerp(a.authority.rl, b.authority.rl, alpha),
        };
        s.trust = new DilTrustState
        {
            human_to_machine = Mathf.Lerp(a.trust.human_to_machine, b.trust.human_to_machine, alpha),
            machine_to_human = Mathf.Lerp(a.trust.machine_to_human, b.trust.machine_to_human, alpha),
        };
        s.risk = new DilRiskState
        {
            environment_urgency = Mathf.Lerp(a.risk.environment_urgency, b.risk.environment_urgency, alpha),
            front_distance_m = Mathf.Lerp(a.risk.front_distance_m, b.risk.front_distance_m, alpha),
            ttc_s = Mathf.Lerp(a.risk.ttc_s, b.risk.ttc_s, alpha),
        };
        s.safety = b.safety;
        s.road = b.road;
        return s;
    }

    DilEgoState InterpolateEgo(DilEgoState a, DilEgoState b, float alpha)
    {
        return new DilEgoState
        {
            x = Mathf.Lerp(a.x, b.x, alpha),
            y = Mathf.Lerp(a.y, b.y, alpha),
            yaw = LerpAngleRad(a.yaw, b.yaw, alpha),
            speed = Mathf.Lerp(a.speed, b.speed, alpha),
            acceleration = Mathf.Lerp(a.acceleration, b.acceleration, alpha),
            steer = Mathf.Lerp(a.steer, b.steer, alpha),
            length = b.length,
            width = b.width,
        };
    }

    DilVehicleState[] InterpolateVehicles(DilVehicleState[] a, DilVehicleState[] b, float alpha)
    {
        if (b == null) return a;
        if (a == null) return b;
        Dictionary<string, DilVehicleState> previous = new Dictionary<string, DilVehicleState>();
        foreach (DilVehicleState item in a)
        {
            previous[VehicleKey(item)] = item;
        }

        DilVehicleState[] result = new DilVehicleState[b.Length];
        for (int i = 0; i < b.Length; ++i)
        {
            DilVehicleState current = b[i];
            if (!previous.TryGetValue(VehicleKey(current), out DilVehicleState prev))
            {
                result[i] = current;
                continue;
            }
            result[i] = new DilVehicleState
            {
                id = current.id,
                slot = current.slot,
                name = current.name,
                x = Mathf.Lerp(prev.x, current.x, alpha),
                y = Mathf.Lerp(prev.y, current.y, alpha),
                yaw = LerpAngleRad(prev.yaw, current.yaw, alpha),
                length = current.length,
                width = current.width,
            };
        }
        return result;
    }

    string VehicleKey(DilVehicleState state)
    {
        return string.IsNullOrEmpty(state.id) ? state.slot.ToString() : state.id;
    }

    float LerpAngleRad(float a, float b, float alpha)
    {
        float deg = Mathf.LerpAngle(a * Mathf.Rad2Deg, b * Mathf.Rad2Deg, alpha);
        return deg * Mathf.Deg2Rad;
    }

    void OnDestroy()
    {
        running = false;
        udp?.Close();
        if (receiveThread != null && receiveThread.IsAlive)
        {
            receiveThread.Join(200);
        }
    }
}
