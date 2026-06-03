using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public static class CreateDilScene
{
    [MenuItem("TASE/Create Driver-in-the-Loop Scene")]
    public static void CreateScene()
    {
        Directory.CreateDirectory("Assets/Scenes");
        Directory.CreateDirectory("Assets/Materials");

        Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        scene.name = "DILHighway";

        Material road = CreateMaterial("Assets/Materials/DIL_Road.mat", new Color(0.18f, 0.21f, 0.25f));
        Material lane = CreateMaterial("Assets/Materials/DIL_Lane.mat", new Color(0.90f, 0.94f, 0.98f));
        Material ego = CreateMaterial("Assets/Materials/DIL_Ego.mat", new Color(0.93f, 0.16f, 0.16f));
        Material surrounding = CreateMaterial("Assets/Materials/DIL_Surrounding.mat", new Color(0.68f, 0.74f, 0.82f));

        GameObject client = new GameObject("DIL_Client");
        DilUdpClient udp = client.AddComponent<DilUdpClient>();
        udp.useRenderInterpolation = true;
        udp.renderDelaySeconds = 0.06f;
        udp.maxBufferedStates = 240;
        LogitechForceFeedbackController forceFeedback = client.AddComponent<LogitechForceFeedbackController>();
        forceFeedback.udpClient = udp;
        forceFeedback.enableForceFeedback = false;
        UnityDriverInputSender inputSender = client.AddComponent<UnityDriverInputSender>();
        inputSender.maxSteerRad = 0.14f;
        inputSender.steerRiseRate = 1.4f;
        inputSender.steerReturnRate = 2.2f;
        HighwaySceneController controller = client.AddComponent<HighwaySceneController>();
        controller.udpClient = udp;
        controller.roadMaterial = road;
        controller.laneMaterial = lane;
        controller.egoMaterial = ego;
        controller.surroundingMaterial = surrounding;
        controller.defaultEgoVehiclePrefabName = "PaperCarsCabrio2DayRed Variant";
        controller.defaultSurroundingVehiclePrefabName = "PaperCarsCabrio2NightWhite Variant";
        controller.cameraMode = 0;
        controller.applyRuntimeValidationPreset = true;
        controller.vehicleHeight = 1.55f;
        controller.roadLengthAhead = 320.0f;
        controller.roadLengthBehind = 90.0f;
        controller.fixedRoadInWorld = true;
        controller.useVisualSmoothing = false;
        controller.visualFollowRate = 14.0f;
        controller.smoothEgoForCamera = false;
        controller.egoCameraFollowRate = 18.0f;
        controller.cameraHeight = 13.0f;
        controller.cameraBackDistance = 28.0f;
        controller.cameraLookAheadDistance = 44.0f;
        controller.cameraLookHeight = 1.0f;
        controller.overheadCameraHeight = 72.0f;
        controller.cameraPositionSmoothTime = 0.45f;
        controller.cameraRotationLerp = 0.08f;
        controller.overheadSizeSmoothTime = 0.65f;
        controller.stableOverheadCamera = true;
        controller.driverCameraHeight = 1.26f;
        controller.driverCameraForwardOffset = 0.62f;
        controller.driverCameraLookAhead = 76.0f;
        controller.driverCameraLookHeight = 1.18f;
        controller.hideEgoInDriverView = true;
        controller.driverCameraSmoothTime = 0.0f;
        controller.driverCameraRotationFollowRate = 90.0f;
        controller.lockDriverCameraToRoad = false;
        controller.showRearViewMirrors = true;
        controller.showCockpitOverlay = false;
        controller.showCockpitModel = false;
        controller.mirrorCameraHeight = 1.48f;
        controller.mirrorCameraBackOffset = -0.10f;
        controller.mirrorCameraLateralOffset = 1.18f;
        controller.mirrorCameraYawDeg = 18.0f;
        controller.centerMirrorYawDeg = 0.0f;
        controller.mirrorCameraFov = 52.0f;

        GameObject cameraObject = new GameObject("Driver_Follow_Camera");
        Camera camera = cameraObject.AddComponent<Camera>();
        camera.clearFlags = CameraClearFlags.Skybox;
        camera.fieldOfView = 65.0f;
        camera.nearClipPlane = 0.05f;
        camera.farClipPlane = 600.0f;
        cameraObject.tag = "MainCamera";
        controller.followCamera = camera;

        GameObject lightObject = new GameObject("Directional Light");
        Light light = lightObject.AddComponent<Light>();
        light.type = LightType.Directional;
        light.intensity = 1.0f;
        lightObject.transform.rotation = Quaternion.Euler(50.0f, -30.0f, 0.0f);

        Text hud = CreateHud();
        controller.hudText = hud;

        string scenePath = "Assets/Scenes/DILHighway.unity";
        EditorSceneManager.SaveScene(scene, scenePath);
        EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(scenePath, true) };
        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();
        Selection.activeGameObject = client;
        Debug.Log("Created DIL scene at " + scenePath + ". Press Play, then start the Python backend.");
    }

    private static Material CreateMaterial(string path, Color color)
    {
        Material mat = AssetDatabase.LoadAssetAtPath<Material>(path);
        if (mat == null)
        {
            mat = new Material(Shader.Find("Standard"));
            AssetDatabase.CreateAsset(mat, path);
        }
        mat.color = color;
        EditorUtility.SetDirty(mat);
        return mat;
    }

    private static Text CreateHud()
    {
        GameObject canvasObject = new GameObject("HUD Canvas");
        Canvas canvas = canvasObject.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        CanvasScaler scaler = canvasObject.AddComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1920, 1080);
        canvasObject.AddComponent<GraphicRaycaster>();

        GameObject textObject = new GameObject("HUD Text");
        textObject.transform.SetParent(canvasObject.transform, false);
        Text text = textObject.AddComponent<Text>();
        text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        if (text.font == null)
        {
            text.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
        }
        text.fontSize = 22;
        text.color = Color.white;
        text.alignment = TextAnchor.UpperLeft;
        text.text = "Waiting for Python DIL backend...";

        RectTransform rect = text.GetComponent<RectTransform>();
        rect.anchorMin = new Vector2(0.0f, 1.0f);
        rect.anchorMax = new Vector2(0.0f, 1.0f);
        rect.pivot = new Vector2(0.0f, 1.0f);
        rect.anchoredPosition = new Vector2(24.0f, -24.0f);
        rect.sizeDelta = new Vector2(520.0f, 260.0f);
        return text;
    }
}
