using UnityEngine;

public class LogitechForceFeedbackController : MonoBehaviour
{
    public DilUdpClient udpClient;
    public bool enableForceFeedback = false;
    public int wheelIndex = 0;
    public int baseSpringPercent = 18;
    public int baseDamperPercent = 14;
    public int maxSpringPercent = 58;
    public int maxDamperPercent = 48;
    public int maxGuidanceForcePercent = 28;
    public bool centerSpringOnly = true;
    public bool enableDamper = true;
    public float trustSensitivity = 0.55f;
    public float riskSensitivity = 0.75f;
    public float steeringConflictScale = 10.0f;
    public bool playHardwareTestForce = false;
    public int hardwareTestForcePercent = 20;
    public string status = "Disabled";

    private bool warnedNoSdk;

    void Start()
    {
        if (udpClient == null) udpClient = GetComponent<DilUdpClient>();
#if LOGITECH_STEERING_WHEEL_SDK
        try
        {
            LogitechGSDK.LogiSteeringInitialize(false);
        }
        catch (System.DllNotFoundException ex)
        {
            status = "Logitech wrapper DLL failed to load. Use a Release x64 SDK DLL.";
            Debug.LogWarning($"Logitech force feedback unavailable: {ex.Message}");
        }
#endif
    }

    void Update()
    {
        if (!enableForceFeedback)
        {
            status = "Disabled. Check Enable Force Feedback to activate.";
            StopForces();
            return;
        }

        if (playHardwareTestForce)
        {
            ApplyHardwareTestForce();
            return;
        }

        if (udpClient == null || !udpClient.TryGetState(out DilSimState state) || state == null)
        {
            status = "Waiting for DIL UDP state.";
            StopForces();
            return;
        }

        ApplyForces(state);
    }

    void OnDisable()
    {
        StopForces();
    }

    void OnApplicationQuit()
    {
        StopForces();
#if LOGITECH_STEERING_WHEEL_SDK
        try
        {
            LogitechGSDK.LogiSteeringShutdown();
        }
        catch (System.DllNotFoundException)
        {
            status = "Logitech wrapper DLL failed to load. Use a Release x64 SDK DLL.";
        }
#endif
    }

    private void ApplyForces(DilSimState state)
    {
        float authority = state.authority != null ? Mathf.Clamp01(state.authority.rl) : 0.5f;
        float risk = state.risk != null ? Mathf.Clamp01(state.risk.environment_urgency) : 0.0f;
        float trustHuman = state.trust != null ? Mathf.Clamp01(state.trust.human_to_machine) : 0.5f;
        float machineSteer = state.ego != null ? state.ego.steer : 0.0f;
        float driverSteer = state.driver_input != null ? state.driver_input.delta_rad : machineSteer;
        float conflict = Mathf.Clamp01(Mathf.Abs(machineSteer - driverSteer) * steeringConflictScale);

        int spring = Mathf.RoundToInt(Mathf.Clamp(
            baseSpringPercent + 100.0f * (riskSensitivity * risk + trustSensitivity * (1.0f - trustHuman) * authority),
            baseSpringPercent,
            maxSpringPercent
        ));
        int damper = Mathf.RoundToInt(Mathf.Clamp(
            baseDamperPercent + 100.0f * (0.35f * risk + 0.45f * conflict),
            baseDamperPercent,
            maxDamperPercent
        ));
        int guidance = Mathf.RoundToInt(Mathf.Clamp(
            100.0f * authority * conflict,
            0.0f,
            maxGuidanceForcePercent
        ));
        if (machineSteer < driverSteer) guidance = -guidance;
        if (centerSpringOnly) guidance = 0;
        if (!enableDamper) damper = 0;

        ApplyLogitechForces(spring, damper, guidance);
    }

    private void ApplyLogitechForces(int springPercent, int damperPercent, int guidancePercent)
    {
#if LOGITECH_STEERING_WHEEL_SDK
        try
        {
            LogitechGSDK.LogiUpdate();
            if (!LogitechGSDK.LogiIsConnected(wheelIndex))
            {
                status = "SDK loaded, but no Logitech wheel is connected.";
                return;
            }
            LogitechGSDK.LogiStopConstantForce(wheelIndex);
            LogitechGSDK.LogiPlaySpringForce(wheelIndex, 0, Mathf.Clamp(springPercent, 0, 100), 45);
            if (damperPercent > 0)
            {
                LogitechGSDK.LogiPlayDamperForce(wheelIndex, Mathf.Clamp(damperPercent, 0, 100));
            }
            else
            {
                LogitechGSDK.LogiStopDamperForce(wheelIndex);
            }
            if (!centerSpringOnly && guidancePercent != 0)
            {
                LogitechGSDK.LogiPlayConstantForce(wheelIndex, Mathf.Clamp(guidancePercent, -100, 100));
            }
            status = $"Active: spring={springPercent}%, damper={damperPercent}%, guidance={guidancePercent}%";
        }
        catch (System.DllNotFoundException ex)
        {
            status = "Logitech wrapper DLL failed to load. Use a Release x64 SDK DLL.";
            if (!warnedNoSdk)
            {
                warnedNoSdk = true;
                Debug.LogWarning($"Logitech force feedback unavailable: {ex.Message}");
            }
        }
#else
        status = "SDK missing. Import Logitech SDK and add LOGITECH_STEERING_WHEEL_SDK.";
        if (!warnedNoSdk)
        {
            warnedNoSdk = true;
            Debug.LogWarning(
                "Logitech force feedback is disabled because LOGITECH_STEERING_WHEEL_SDK is not defined. " +
                "Import the Logitech Steering Wheel SDK, then enable this scripting define."
            );
        }
#endif
    }

    private void ApplyHardwareTestForce()
    {
#if LOGITECH_STEERING_WHEEL_SDK
        try
        {
            LogitechGSDK.LogiUpdate();
            if (!LogitechGSDK.LogiIsConnected(wheelIndex))
            {
                status = "Hardware test failed: no Logitech wheel is connected.";
                return;
            }
            int force = Mathf.Clamp(Mathf.Abs(hardwareTestForcePercent), 0, 100);
            LogitechGSDK.LogiStopConstantForce(wheelIndex);
            LogitechGSDK.LogiPlaySpringForce(wheelIndex, 0, force, 45);
            LogitechGSDK.LogiPlayDamperForce(wheelIndex, force);
            status = $"Hardware test active: spring={force}%, damper={force}%";
        }
        catch (System.DllNotFoundException ex)
        {
            status = "Logitech wrapper DLL failed to load. Use a Release x64 SDK DLL.";
            if (!warnedNoSdk)
            {
                warnedNoSdk = true;
                Debug.LogWarning($"Logitech hardware test unavailable: {ex.Message}");
            }
        }
#else
        status = "Hardware test unavailable: Logitech SDK is not imported/defined.";
        if (!warnedNoSdk)
        {
            warnedNoSdk = true;
            Debug.LogWarning(
                "Logitech force feedback hardware test is unavailable because LOGITECH_STEERING_WHEEL_SDK is not defined."
            );
        }
#endif
    }

    private void StopForces()
    {
#if LOGITECH_STEERING_WHEEL_SDK
        try
        {
            LogitechGSDK.LogiStopSpringForce(wheelIndex);
            LogitechGSDK.LogiStopDamperForce(wheelIndex);
            LogitechGSDK.LogiStopConstantForce(wheelIndex);
        }
        catch (System.DllNotFoundException)
        {
            status = "Logitech wrapper DLL failed to load. Use a Release x64 SDK DLL.";
        }
#endif
    }
}
