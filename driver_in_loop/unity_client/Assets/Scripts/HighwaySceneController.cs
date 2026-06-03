using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
#if UNITY_EDITOR
using UnityEditor;
#endif

public class HighwaySceneController : MonoBehaviour
{
    public DilUdpClient udpClient;
    public Camera followCamera;
    public Material roadMaterial;
    public Material laneMaterial;
    public Material egoMaterial;
    public Material surroundingMaterial;
    public GameObject egoVehiclePrefab;
    public GameObject surroundingVehiclePrefab;
    public string defaultEgoVehiclePrefabName = "PaperCarsCabrio2DayRed Variant";
    public string defaultSurroundingVehiclePrefabName = "PaperCarsCabrio2NightWhite Variant";
    public bool useImportedVehiclePrefabs = true;
    public bool tintImportedVehiclePrefabs = false;
    public Vector3 importedVehicleEulerOffset = new Vector3(0.0f, 90.0f, 0.0f);
    public bool showMachineIntention = true;
    public float intentionLineWidth = 0.16f;
    public Text hudText;
    public Text riskWarningText;
    public float conflictWarningThresholdRad = 0.09f;
    public float riskWarningThreshold = 0.45f;
    public float warningBlinkFrequencyHz = 3.0f;
    public bool showRearViewMirrors = true;
    public bool showCockpitOverlay = false;
    public bool showCockpitModel = false;
    public int mirrorTextureWidth = 320;
    public int mirrorTextureHeight = 130;
    public int centerMirrorTextureWidth = 500;
    public int centerMirrorTextureHeight = 125;
    public float mirrorCameraHeight = 1.48f;
    public float mirrorCameraBackOffset = -0.10f;
    public float mirrorCameraLateralOffset = 1.18f;
    public float mirrorCameraYawDeg = 18.0f;
    public float centerMirrorYawDeg = 0.0f;
    public float mirrorCameraFov = 52.0f;

    public float positionScale = 1.0f;
    public float roadLengthAhead = 320.0f;
    public float roadLengthBehind = 90.0f;
    public float laneLineWidth = 0.08f;
    public float dashedLaneLength = 6.0f;
    public float dashedLaneGap = 8.0f;
    public bool fixedRoadInWorld = true;
    public bool useVisualSmoothing = false;
    public float visualFollowRate = 14.0f;
    public bool smoothEgoForCamera = false;
    public float egoCameraFollowRate = 18.0f;
    public float vehicleHeight = 1.55f;
    public int cameraMode = 1;
    public float cameraHeight = 10.0f;
    public float cameraBackDistance = 22.0f;
    public float cameraLookAheadDistance = 36.0f;
    public float cameraLookHeight = 1.0f;
    public float cameraLateralOffset = 0.0f;
    public float driverCameraHeight = 1.26f;
    public float driverCameraForwardOffset = 0.62f;
    public float driverCameraLookAhead = 76.0f;
    public float driverCameraLookHeight = 1.18f;
    public bool hideEgoInDriverView = true;
    public float driverCameraSmoothTime = 0.0f;
    public float driverCameraRotationFollowRate = 90.0f;
    public bool lockDriverCameraToRoad = false;
    public float overheadCameraHeight = 58.0f;
    public bool applyRuntimeValidationPreset = true;
    public float cameraPositionSmoothTime = 0.45f;
    public float cameraRotationLerp = 0.08f;
    public float overheadSizeSmoothTime = 0.65f;
    public bool stableOverheadCamera = true;

    private GameObject ego;
    private GameObject runtimeRoot;
    private GameObject driverCameraRig;
    private readonly Dictionary<string, GameObject> vehicles = new Dictionary<string, GameObject>();
    private readonly HashSet<GameObject> visualInitialized = new HashSet<GameObject>();
    private readonly List<GameObject> roadObjects = new List<GameObject>();
    private float originX;
    private bool originReady;
    private Vector3 cameraVelocity;
    private float overheadSizeVelocity;
    private bool overheadFrameReady;
    private Vector3 overheadFrameCenter;
    private float overheadFrameSize = 35.0f;
    private int lastFrameIndex = -1;
    private float roadCenterZ;
    private Vector3 driverCameraVelocity;
    private bool driverCameraReady;
    private DilSimState renderCameraState;
    private Material windowMaterial;
    private Material tireMaterial;
    private Material headlightMaterial;
    private Material taillightMaterial;
    private Material intentionMaterial;
    private Material cockpitMaterial;
    private Material cockpitTrimMaterial;
    private LineRenderer machineIntentionLine;
    private Camera leftMirrorCamera;
    private Camera rightMirrorCamera;
    private Camera centerMirrorCamera;
    private RawImage leftMirrorImage;
    private RawImage rightMirrorImage;
    private RawImage centerMirrorImage;
    private RectTransform steeringWheelOverlay;
    private GameObject cockpitRoot;
    private Transform cockpitWheelRoot;
    private RenderTexture leftMirrorTexture;
    private RenderTexture rightMirrorTexture;
    private RenderTexture centerMirrorTexture;
    private float warningVisibleUntil;

    void Start()
    {
        if (udpClient == null) udpClient = GetComponent<DilUdpClient>();
        if (followCamera == null) followCamera = Camera.main;
        AutoAssignVehiclePrefabs();
        EnsureMaterials();
        ApplyValidationPreset();
        EnsureRuntimeRoot();
        HideLegacyGeneratedObjects();
        ClearRuntimeRootChildren();
        EnsureHudCanvas();
        EnsureDriverCameraRig();
        EnsureRearViewMirrors();
        EnsureRiskWarningText();
        EnsureCockpitOverlay();
        EnsureCockpitModel();
        BringDriverUiToFront();
        ego = CreateVehicle("ego", egoMaterial, true);
        DeactivateDuplicateEgos();
    }

    void AutoAssignVehiclePrefabs()
    {
#if UNITY_EDITOR
        if (egoVehiclePrefab == null && defaultEgoVehiclePrefabName.Length > 0)
        {
            egoVehiclePrefab = FindPrefabByName(defaultEgoVehiclePrefabName);
        }
        if (surroundingVehiclePrefab == null && defaultSurroundingVehiclePrefabName.Length > 0)
        {
            surroundingVehiclePrefab = FindPrefabByName(defaultSurroundingVehiclePrefabName);
        }
#endif
    }

#if UNITY_EDITOR
    GameObject FindPrefabByName(string prefabName)
    {
        string[] guids = AssetDatabase.FindAssets(prefabName + " t:Prefab");
        foreach (string guid in guids)
        {
            string path = AssetDatabase.GUIDToAssetPath(guid);
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);
            if (prefab != null && prefab.name == prefabName)
            {
                return prefab;
            }
        }
        return null;
    }
#endif

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.C)) cameraMode = (cameraMode + 1) % 3;
        if (udpClient == null || !udpClient.TryGetRenderState(out DilSimState state)) return;
        if (!originReady)
        {
            originX = state.ego.x;
            originReady = true;
            overheadFrameReady = false;
            visualInitialized.Clear();
            BuildRoad(state);
        }
        if (state.frame_index < lastFrameIndex)
        {
            overheadFrameReady = false;
            visualInitialized.Clear();
            driverCameraReady = false;
        }
        lastFrameIndex = state.frame_index;

        UpdateVehicle(ego, state.ego.x, state.ego.y, state.ego.yaw, state.ego.length, state.ego.width);
        DeactivateDuplicateEgos();
        SyncSurroundingVehicles(state);
        UpdateMachineIntention(state);
        if (!fixedRoadInWorld) UpdateRoadPosition(state.ego.x);
        renderCameraState = state;
        UpdateHud(state);
        UpdateRiskWarning(state);
        UpdateCockpitOverlay(state);
        UpdateCockpitModel(state);
    }

    void LateUpdate()
    {
        if (renderCameraState != null)
        {
            UpdateCamera(renderCameraState);
            UpdateRearViewMirrors(renderCameraState);
        }
    }

    void EnsureMaterials()
    {
        if (roadMaterial == null) roadMaterial = MakeMaterial(new Color(0.18f, 0.21f, 0.25f));
        if (laneMaterial == null) laneMaterial = MakeMaterial(new Color(0.90f, 0.94f, 0.98f));
        if (egoMaterial == null) egoMaterial = MakeMaterial(new Color(0.93f, 0.16f, 0.16f));
        if (surroundingMaterial == null) surroundingMaterial = MakeMaterial(new Color(0.68f, 0.74f, 0.82f));
        surroundingMaterial.color = new Color(0.68f, 0.74f, 0.82f);
        if (windowMaterial == null) windowMaterial = MakeMaterial(new Color(0.12f, 0.18f, 0.24f, 0.92f));
        if (tireMaterial == null) tireMaterial = MakeMaterial(new Color(0.025f, 0.025f, 0.025f));
        if (headlightMaterial == null) headlightMaterial = MakeMaterial(new Color(1.00f, 0.90f, 0.55f));
        if (taillightMaterial == null) taillightMaterial = MakeMaterial(new Color(0.75f, 0.02f, 0.02f));
        if (intentionMaterial == null) intentionMaterial = MakeMaterial(new Color(1.0f, 0.63f, 0.12f, 0.82f));
        if (cockpitMaterial == null) cockpitMaterial = MakeMaterial(new Color(0.018f, 0.020f, 0.024f, 1.0f));
        if (cockpitTrimMaterial == null) cockpitTrimMaterial = MakeMaterial(new Color(0.60f, 0.65f, 0.72f, 1.0f));
    }

    void ApplyValidationPreset()
    {
        if (!applyRuntimeValidationPreset) return;
        cameraMode = 0;
        vehicleHeight = 1.55f;
        cameraHeight = 13.0f;
        cameraBackDistance = 28.0f;
        cameraLookAheadDistance = 44.0f;
        cameraLookHeight = 1.0f;
        overheadCameraHeight = 72.0f;
        roadLengthAhead = 900.0f;
        roadLengthBehind = 260.0f;
        laneLineWidth = 0.10f;
        dashedLaneLength = 6.0f;
        dashedLaneGap = 8.0f;
        fixedRoadInWorld = true;
        useVisualSmoothing = false;
        visualFollowRate = 18.0f;
        smoothEgoForCamera = false;
        egoCameraFollowRate = 22.0f;
        cameraPositionSmoothTime = 0.45f;
        cameraRotationLerp = 0.08f;
        overheadSizeSmoothTime = 0.65f;
        stableOverheadCamera = true;
        driverCameraHeight = 1.26f;
        driverCameraForwardOffset = 0.62f;
        driverCameraLookAhead = 76.0f;
        driverCameraLookHeight = 1.18f;
        hideEgoInDriverView = true;
        driverCameraSmoothTime = 0.0f;
        driverCameraRotationFollowRate = 90.0f;
        lockDriverCameraToRoad = false;
        showRearViewMirrors = true;
        showCockpitOverlay = false;
        showCockpitModel = false;
        mirrorCameraHeight = 1.48f;
        mirrorCameraBackOffset = -0.10f;
        mirrorCameraLateralOffset = 1.18f;
        mirrorCameraYawDeg = 18.0f;
        centerMirrorYawDeg = 0.0f;
        mirrorCameraFov = 52.0f;
        if (udpClient != null)
        {
            udpClient.useRenderInterpolation = true;
            udpClient.renderDelaySeconds = 0.06f;
            udpClient.maxBufferedStates = 240;
        }
    }

    Material MakeMaterial(Color color)
    {
        Material mat = new Material(Shader.Find("Standard"));
        mat.color = color;
        return mat;
    }

    GameObject CreateVehicle(string name, Material material, bool isEgo)
    {
        GameObject root = new GameObject(name);
        if (runtimeRoot != null) root.transform.SetParent(runtimeRoot.transform, true);

        GameObject prefab = isEgo ? egoVehiclePrefab : surroundingVehiclePrefab;
        if (useImportedVehiclePrefabs && prefab != null)
        {
            GameObject model = Instantiate(prefab, root.transform);
            model.name = "model";
            model.transform.localPosition = Vector3.zero;
            model.transform.localRotation = Quaternion.identity;
            model.transform.localScale = Vector3.one;
            RemoveColliders(model);
            if (tintImportedVehiclePrefabs) ApplyMaterialToRenderers(model, material);
            SetVehicleDimensions(root, 4.6f, 1.8f);
            return root;
        }

        GameObject body = CreateVehiclePart(root, "body", PrimitiveType.Cube, material);
        GameObject hood = CreateVehiclePart(root, "hood", PrimitiveType.Cube, material);
        GameObject trunk = CreateVehiclePart(root, "trunk", PrimitiveType.Cube, material);
        GameObject cabin = CreateVehiclePart(root, "cabin", PrimitiveType.Cube, material);
        GameObject windshield = CreateVehiclePart(root, "windshield", PrimitiveType.Cube, windowMaterial);
        GameObject rearWindow = CreateVehiclePart(root, "rear_window", PrimitiveType.Cube, windowMaterial);
        GameObject sideWindowL = CreateVehiclePart(root, "side_window_l", PrimitiveType.Cube, windowMaterial);
        GameObject sideWindowR = CreateVehiclePart(root, "side_window_r", PrimitiveType.Cube, windowMaterial);
        GameObject nose = CreateVehiclePart(root, "front_nose", PrimitiveType.Cube, material);
        GameObject headlightL = CreateVehiclePart(root, "headlight_l", PrimitiveType.Cube, headlightMaterial);
        GameObject headlightR = CreateVehiclePart(root, "headlight_r", PrimitiveType.Cube, headlightMaterial);
        GameObject taillightL = CreateVehiclePart(root, "taillight_l", PrimitiveType.Cube, taillightMaterial);
        GameObject taillightR = CreateVehiclePart(root, "taillight_r", PrimitiveType.Cube, taillightMaterial);
        GameObject[] wheels =
        {
            CreateVehiclePart(root, "wheel_fl", PrimitiveType.Cylinder, tireMaterial),
            CreateVehiclePart(root, "wheel_fr", PrimitiveType.Cylinder, tireMaterial),
            CreateVehiclePart(root, "wheel_rl", PrimitiveType.Cylinder, tireMaterial),
            CreateVehiclePart(root, "wheel_rr", PrimitiveType.Cylinder, tireMaterial),
        };

        foreach (GameObject wheel in wheels)
        {
            wheel.transform.localRotation = Quaternion.Euler(90.0f, 0.0f, 0.0f);
        }

        SetVehicleDimensions(root, 4.6f, 1.8f);
        return root;
    }

    void RemoveColliders(GameObject obj)
    {
        foreach (Component component in obj.GetComponentsInChildren<Component>(true))
        {
            if (component != null && component.GetType().Name.Contains("Collider"))
            {
                Destroy(component);
            }
        }
    }

    void ApplyMaterialToRenderers(GameObject obj, Material material)
    {
        foreach (Renderer renderer in obj.GetComponentsInChildren<Renderer>(true))
        {
            renderer.material = material;
        }
    }

    GameObject CreateVehiclePart(GameObject parent, string name, PrimitiveType primitive, Material material)
    {
        GameObject part = GameObject.CreatePrimitive(primitive);
        part.name = name;
        part.transform.SetParent(parent.transform, false);
        part.GetComponent<Renderer>().material = material;
        Component collider = part.GetComponent("Collider");
        if (collider != null) Destroy(collider);
        return part;
    }

    void EnsureRuntimeRoot()
    {
        runtimeRoot = GameObject.Find("DIL_Runtime");
        if (runtimeRoot == null)
        {
            runtimeRoot = new GameObject("DIL_Runtime");
        }
    }

    void ClearRuntimeRootChildren()
    {
        if (runtimeRoot == null) return;
        for (int i = runtimeRoot.transform.childCount - 1; i >= 0; --i)
        {
            Destroy(runtimeRoot.transform.GetChild(i).gameObject);
        }
        vehicles.Clear();
        roadObjects.Clear();
        visualInitialized.Clear();
    }

    void HideLegacyGeneratedObjects()
    {
        foreach (GameObject obj in GameObject.FindObjectsOfType<GameObject>())
        {
            if (obj == null || obj == gameObject || obj.name == "DIL_Runtime") continue;
            if (obj.transform.parent != null && obj.transform.parent.name == "DIL_Runtime") continue;
            if (obj.name == "ego" || obj.name == "Road" || obj.name == "LaneMark" || obj.name.StartsWith("surrounding_"))
            {
                obj.SetActive(false);
            }
        }
    }

    void DeactivateDuplicateEgos()
    {
        foreach (GameObject obj in GameObject.FindObjectsOfType<GameObject>())
        {
            if (obj == null || obj == ego) continue;
            if (obj.name == "ego" && obj.activeInHierarchy)
            {
                obj.SetActive(false);
            }
        }
    }

    void EnsureDriverCameraRig()
    {
        driverCameraRig = GameObject.Find("Driver_Camera_Rig");
        if (driverCameraRig == null) driverCameraRig = new GameObject("Driver_Camera_Rig");
    }

    void UpdateVehicle(GameObject obj, float x, float y, float yaw, float length, float width)
    {
        Vector3 pos = ToUnityPosition(x, y);
        pos.y = 0.0f;
        Quaternion rot = Quaternion.Euler(0.0f, yaw * Mathf.Rad2Deg, 0.0f);

        SetVehicleDimensions(obj, length, width);

        bool smoothThisObject = useVisualSmoothing || (smoothEgoForCamera && obj == ego);
        float followRate = obj == ego ? egoCameraFollowRate : visualFollowRate;

        if (!smoothThisObject)
        {
            obj.transform.SetPositionAndRotation(pos, rot);
            visualInitialized.Add(obj);
            return;
        }

        // Only snap on the first frame or when a real scene reset / teleport happens.
        if (!visualInitialized.Contains(obj) || Vector3.Distance(obj.transform.position, pos) > 80.0f)
        {
            obj.transform.SetPositionAndRotation(pos, rot);
            visualInitialized.Add(obj);
            return;
        }

        float alpha = 1.0f - Mathf.Exp(-followRate * Time.deltaTime);
        obj.transform.position = Vector3.Lerp(obj.transform.position, pos, alpha);
        obj.transform.rotation = Quaternion.Slerp(obj.transform.rotation, rot, alpha);
    }

    void SetVehicleDimensions(GameObject obj, float length, float width)
    {
        float l = Mathf.Max(length, 3.8f);
        float w = Mathf.Max(width, 1.55f);
        float h = Mathf.Max(vehicleHeight, 1.25f);

        Transform importedModel = obj.transform.Find("model");
        if (importedModel != null)
        {
            FitImportedVehicleModel(importedModel, l, w, h);
            return;
        }

        Transform body = obj.transform.Find("body");
        if (body != null)
        {
            body.localPosition = new Vector3(-l * 0.03f, h * 0.34f, 0.0f);
            body.localScale = new Vector3(l * 0.70f, h * 0.42f, w * 0.95f);
        }

        Transform hood = obj.transform.Find("hood");
        if (hood != null)
        {
            hood.localPosition = new Vector3(l * 0.31f, h * 0.36f, 0.0f);
            hood.localScale = new Vector3(l * 0.28f, h * 0.34f, w * 0.86f);
        }

        Transform trunk = obj.transform.Find("trunk");
        if (trunk != null)
        {
            trunk.localPosition = new Vector3(-l * 0.41f, h * 0.36f, 0.0f);
            trunk.localScale = new Vector3(l * 0.20f, h * 0.34f, w * 0.88f);
        }

        Transform cabin = obj.transform.Find("cabin");
        if (cabin != null)
        {
            cabin.localPosition = new Vector3(-l * 0.08f, h * 0.76f, 0.0f);
            cabin.localScale = new Vector3(l * 0.40f, h * 0.34f, w * 0.66f);
        }

        Transform windshield = obj.transform.Find("windshield");
        if (windshield != null)
        {
            windshield.localPosition = new Vector3(l * 0.16f, h * 0.78f, 0.0f);
            windshield.localScale = new Vector3(l * 0.10f, h * 0.28f, w * 0.68f);
        }

        Transform rearWindow = obj.transform.Find("rear_window");
        if (rearWindow != null)
        {
            rearWindow.localPosition = new Vector3(-l * 0.30f, h * 0.77f, 0.0f);
            rearWindow.localScale = new Vector3(l * 0.09f, h * 0.26f, w * 0.62f);
        }

        Transform sideWindowL = obj.transform.Find("side_window_l");
        if (sideWindowL != null)
        {
            sideWindowL.localPosition = new Vector3(-l * 0.07f, h * 0.82f, w * 0.34f);
            sideWindowL.localScale = new Vector3(l * 0.32f, h * 0.18f, w * 0.035f);
        }

        Transform sideWindowR = obj.transform.Find("side_window_r");
        if (sideWindowR != null)
        {
            sideWindowR.localPosition = new Vector3(-l * 0.07f, h * 0.82f, -w * 0.34f);
            sideWindowR.localScale = new Vector3(l * 0.32f, h * 0.18f, w * 0.035f);
        }

        Transform nose = obj.transform.Find("front_nose");
        if (nose != null)
        {
            nose.localPosition = new Vector3(l * 0.50f, h * 0.36f, 0.0f);
            nose.localScale = new Vector3(l * 0.055f, h * 0.28f, w * 0.74f);
        }

        SetSmallLamp(obj.transform.Find("headlight_l"), l * 0.535f, w * 0.22f, h, true);
        SetSmallLamp(obj.transform.Find("headlight_r"), l * 0.535f, -w * 0.22f, h, true);
        SetSmallLamp(obj.transform.Find("taillight_l"), -l * 0.525f, w * 0.24f, h, false);
        SetSmallLamp(obj.transform.Find("taillight_r"), -l * 0.525f, -w * 0.24f, h, false);

        Transform marker = obj.transform.Find("front_marker");
        if (marker != null)
        {
            marker.localPosition = new Vector3(l * 0.54f, h * 0.58f, 0.0f);
            marker.localScale = new Vector3(l * 0.045f, h * 0.30f, w * 0.24f);
        }

        SetWheel(obj.transform.Find("wheel_fl"), l * 0.32f, w * 0.46f, h, w);
        SetWheel(obj.transform.Find("wheel_fr"), l * 0.32f, -w * 0.46f, h, w);
        SetWheel(obj.transform.Find("wheel_rl"), -l * 0.34f, w * 0.46f, h, w);
        SetWheel(obj.transform.Find("wheel_rr"), -l * 0.34f, -w * 0.46f, h, w);
    }

    void SetWheel(Transform wheel, float x, float z, float h, float w)
    {
        if (wheel == null) return;
        float radius = Mathf.Clamp(w * 0.16f, 0.22f, 0.36f);
        wheel.localPosition = new Vector3(x, radius, z);
        wheel.localRotation = Quaternion.Euler(90.0f, 0.0f, 0.0f);
        wheel.localScale = new Vector3(radius * 2.0f, Mathf.Clamp(w * 0.10f, 0.14f, 0.22f), radius * 2.0f);
    }

    void SetSmallLamp(Transform lamp, float x, float z, float h, bool front)
    {
        if (lamp == null) return;
        lamp.localPosition = new Vector3(x, h * 0.45f, z);
        lamp.localScale = new Vector3(0.06f, h * 0.12f, 0.18f);
        lamp.localRotation = front ? Quaternion.identity : Quaternion.identity;
    }

    void FitImportedVehicleModel(Transform model, float targetLength, float targetWidth, float targetHeight)
    {
        Renderer[] renderers = model.GetComponentsInChildren<Renderer>(true);
        if (renderers.Length == 0)
        {
            model.localScale = Vector3.one;
            model.localPosition = new Vector3(0.0f, targetHeight * 0.5f, 0.0f);
            return;
        }

        model.localRotation = Quaternion.Euler(importedVehicleEulerOffset);
        model.localScale = Vector3.one;
        model.localPosition = Vector3.zero;

        Bounds bounds = renderers[0].bounds;
        for (int i = 1; i < renderers.Length; ++i)
        {
            bounds.Encapsulate(renderers[i].bounds);
        }

        float sx = targetLength / Mathf.Max(bounds.size.x, 0.001f);
        float sz = targetWidth / Mathf.Max(bounds.size.z, 0.001f);
        float sy = targetHeight / Mathf.Max(bounds.size.y, 0.001f);
        float scale = Mathf.Min(sx, Mathf.Min(sz, sy));
        model.localScale = Vector3.one * scale;

        bounds = renderers[0].bounds;
        for (int i = 1; i < renderers.Length; ++i)
        {
            bounds.Encapsulate(renderers[i].bounds);
        }
        Vector3 centerOffset = model.position - bounds.center;
        model.position += new Vector3(centerOffset.x, -bounds.min.y, centerOffset.z);
    }

    Vector3 ToUnityPosition(float x, float y)
    {
        return new Vector3((x - originX) * positionScale, 0.0f, -y * positionScale);
    }

    void SyncSurroundingVehicles(DilSimState state)
    {
        HashSet<string> active = new HashSet<string>();
        if (state.vehicles == null) return;
        foreach (DilVehicleState v in state.vehicles)
        {
            string id = string.IsNullOrEmpty(v.id) ? v.slot.ToString() : v.id;
            active.Add(id);
            if (!vehicles.TryGetValue(id, out GameObject obj))
            {
                obj = CreateVehicle("surrounding_" + id, surroundingMaterial, false);
                vehicles[id] = obj;
            }
            UpdateVehicle(obj, v.x, v.y, v.yaw, v.length, v.width);
        }

        List<string> toRemove = new List<string>();
        foreach (var pair in vehicles)
        {
            if (!active.Contains(pair.Key)) toRemove.Add(pair.Key);
        }
        foreach (string key in toRemove)
        {
            Destroy(vehicles[key]);
            vehicles.Remove(key);
        }
    }

    void UpdateMachineIntention(DilSimState state)
    {
        if (!showMachineIntention || state.intention == null || state.intention.machine == null || state.intention.machine.Length < 2)
        {
            if (machineIntentionLine != null) machineIntentionLine.positionCount = 0;
            return;
        }

        if (machineIntentionLine == null)
        {
            GameObject lineObj = new GameObject("Machine_Intention_Trajectory");
            if (runtimeRoot != null) lineObj.transform.SetParent(runtimeRoot.transform, true);
            machineIntentionLine = lineObj.AddComponent<LineRenderer>();
            machineIntentionLine.material = intentionMaterial;
            machineIntentionLine.useWorldSpace = true;
            machineIntentionLine.numCapVertices = 4;
            machineIntentionLine.numCornerVertices = 4;
            machineIntentionLine.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            machineIntentionLine.receiveShadows = false;
        }

        machineIntentionLine.startWidth = intentionLineWidth;
        machineIntentionLine.endWidth = intentionLineWidth;
        machineIntentionLine.positionCount = state.intention.machine.Length;
        for (int i = 0; i < state.intention.machine.Length; ++i)
        {
            DilPointState p = state.intention.machine[i];
            Vector3 pos = ToUnityPosition(p.x, p.y);
            pos.y = 0.12f;
            machineIntentionLine.SetPosition(i, pos);
        }
    }

    void BuildRoad(DilSimState state)
    {
        foreach (GameObject obj in roadObjects) Destroy(obj);
        roadObjects.Clear();

        float[] marks = state.road != null && state.road.lane_markings != null && state.road.lane_markings.Length >= 2
            ? state.road.lane_markings
            : new float[] { -5.25f, -1.75f, 1.75f, 5.25f };

        float minY = marks[0];
        float maxY = marks[marks.Length - 1];
        float width = Mathf.Abs(maxY - minY) + 3.5f;
        roadCenterZ = -0.5f * (minY + maxY) * positionScale;
        float roadWindowCenterX = (roadLengthAhead - roadLengthBehind) * 0.5f;
        GameObject road = GameObject.CreatePrimitive(PrimitiveType.Cube);
        road.name = "Road";
        if (runtimeRoot != null) road.transform.SetParent(runtimeRoot.transform, true);
        road.GetComponent<Renderer>().material = roadMaterial;
        road.transform.localScale = new Vector3(roadLengthAhead + roadLengthBehind, 0.03f, width);
        road.transform.position = new Vector3(roadWindowCenterX, 0.0f, roadCenterZ);
        roadObjects.Add(road);

        for (int i = 0; i < marks.Length; ++i)
        {
            bool isRoadEdge = i == 0 || i == marks.Length - 1;
            if (isRoadEdge)
            {
                CreateSolidLaneMark(marks[i]);
            }
            else
            {
                CreateDashedLaneMark(marks[i]);
            }
        }
    }

    void CreateSolidLaneMark(float mark)
    {
        GameObject line = GameObject.CreatePrimitive(PrimitiveType.Cube);
        line.name = "LaneMark_Solid";
        if (runtimeRoot != null) line.transform.SetParent(runtimeRoot.transform, true);
        line.GetComponent<Renderer>().material = laneMaterial;
        line.transform.localScale = new Vector3(roadLengthAhead + roadLengthBehind, 0.018f, laneLineWidth);
        line.transform.position = new Vector3((roadLengthAhead - roadLengthBehind) * 0.5f, 0.075f, -mark * positionScale);
        roadObjects.Add(line);
    }

    void CreateDashedLaneMark(float mark)
    {
        float startX = -roadLengthBehind;
        float endX = roadLengthAhead;
        float step = Mathf.Max(0.1f, dashedLaneLength + dashedLaneGap);
        for (float x = startX; x <= endX; x += step)
        {
            float segLength = Mathf.Min(dashedLaneLength, endX - x);
            if (segLength <= 0.05f) continue;
            GameObject dash = GameObject.CreatePrimitive(PrimitiveType.Cube);
            dash.name = "LaneMark_Dash";
            if (runtimeRoot != null) dash.transform.SetParent(runtimeRoot.transform, true);
            dash.GetComponent<Renderer>().material = laneMaterial;
            dash.transform.localScale = new Vector3(segLength, 0.018f, laneLineWidth);
            dash.transform.position = new Vector3(x + segLength * 0.5f, 0.075f, -mark * positionScale);
            roadObjects.Add(dash);
        }
    }

    void UpdateRoadPosition(float egoX)
    {
        float centerX = (egoX - originX + (roadLengthAhead - roadLengthBehind) * 0.5f) * positionScale;
        foreach (GameObject obj in roadObjects)
        {
            Vector3 p = obj.transform.position;
            p.x = centerX;
            obj.transform.position = p;
        }
    }

    void UpdateCamera(DilSimState state)
    {
        if (followCamera == null) return;
        Vector3 egoPos = ToUnityPosition(state.ego.x, state.ego.y);
        Quaternion yaw = Quaternion.Euler(0.0f, state.ego.yaw * Mathf.Rad2Deg, 0.0f);
        Vector3 roadForward = yaw * Vector3.right;
        Vector3 roadRight = yaw * Vector3.back;
        Vector3 camPos;
        Vector3 lookTarget;
        Vector3 cameraUp = Vector3.up;
        if (cameraMode == 0)
        {
            followCamera.orthographic = false;
            Quaternion egoRotation = lockDriverCameraToRoad ? Quaternion.identity : (ego != null ? ego.transform.rotation : yaw);
            Vector3 egoVisualPos = ego != null ? ego.transform.position : egoPos;
            roadForward = egoRotation * Vector3.right;
            roadRight = egoRotation * Vector3.back;
            camPos =
                egoVisualPos
                + roadForward * driverCameraForwardOffset
                + roadRight * cameraLateralOffset
                + Vector3.up * driverCameraHeight;
            lookTarget =
                egoVisualPos
                + roadForward * driverCameraLookAhead
                + Vector3.up * driverCameraLookHeight;
            followCamera.fieldOfView = 78.0f;
            cameraUp = Vector3.up;
        }
        else if (cameraMode == 2)
        {
            followCamera.orthographic = true;
            if (stableOverheadCamera)
            {
                EnsureStableOverheadFrame(state, egoPos);
                camPos = overheadFrameCenter + Vector3.up * overheadCameraHeight;
                lookTarget = overheadFrameCenter;
                followCamera.orthographicSize = overheadFrameSize;
                cameraUp = Vector3.right;
            }
            else
            {
                roadForward = Vector3.right;
                roadRight = Vector3.back;
                Bounds bounds = CurrentTrafficBounds(egoPos);
                camPos = bounds.center + Vector3.up * overheadCameraHeight;
                lookTarget = bounds.center;
                float aspect = Mathf.Max(0.1f, followCamera.aspect);
                float halfLongitudinal = Mathf.Max(20.0f, bounds.size.x * 0.5f);
                float halfLateral = Mathf.Max(8.0f, bounds.size.z * 0.5f);
                float targetSize = Mathf.Clamp(
                    Mathf.Max(halfLateral, halfLongitudinal / aspect) * 1.25f,
                    18.0f,
                    70.0f
                );
                followCamera.orthographicSize = Mathf.SmoothDamp(
                    followCamera.orthographicSize,
                    targetSize,
                    ref overheadSizeVelocity,
                    overheadSizeSmoothTime
                );
                cameraUp = roadForward;
            }
        }
        else
        {
            followCamera.orthographic = false;
            Vector3 cameraBasePos = ego != null ? ego.transform.position : egoPos;
            Quaternion cameraBaseRot = ego != null ? ego.transform.rotation : yaw;
            roadForward = cameraBaseRot * Vector3.right;
            roadRight = cameraBaseRot * Vector3.back;
            camPos =
                cameraBasePos
                - roadForward * cameraBackDistance
                + roadRight * cameraLateralOffset
                + Vector3.up * cameraHeight;
            lookTarget =
                cameraBasePos
                + roadForward * cameraLookAheadDistance
                + Vector3.up * cameraLookHeight;
            followCamera.fieldOfView = 62.0f;
        }
        Quaternion targetRotation = Quaternion.LookRotation(lookTarget - camPos, cameraUp);
        SetEgoVisible(!(cameraMode == 0 && hideEgoInDriverView));
        if (cameraMode == 0)
        {
            UpdateDriverCameraRig(camPos, targetRotation);
        }
        else if (cameraMode == 2 && stableOverheadCamera)
        {
            followCamera.transform.position = camPos;
            followCamera.transform.rotation = targetRotation;
        }
        else
        {
            followCamera.transform.position = Vector3.SmoothDamp(
                followCamera.transform.position,
                camPos,
                ref cameraVelocity,
                cameraPositionSmoothTime
            );
            followCamera.transform.rotation = Quaternion.Slerp(
                followCamera.transform.rotation,
                targetRotation,
                cameraRotationLerp
            );
        }
    }

    void UpdateDriverCameraRig(Vector3 targetPosition, Quaternion targetRotation)
    {
        if (driverCameraRig == null) EnsureDriverCameraRig();
        if (!driverCameraReady || driverCameraSmoothTime <= 0.0f || Vector3.Distance(driverCameraRig.transform.position, targetPosition) > 12.0f)
        {
            driverCameraRig.transform.position = targetPosition;
            driverCameraRig.transform.rotation = targetRotation;
            driverCameraVelocity = Vector3.zero;
            driverCameraReady = true;
        }
        else
        {
            driverCameraRig.transform.position = Vector3.SmoothDamp(
                driverCameraRig.transform.position,
                targetPosition,
                ref driverCameraVelocity,
                driverCameraSmoothTime
            );
            float alpha = 1.0f - Mathf.Exp(-driverCameraRotationFollowRate * Time.deltaTime);
            driverCameraRig.transform.rotation = Quaternion.Slerp(driverCameraRig.transform.rotation, targetRotation, alpha);
        }

        followCamera.transform.position = driverCameraRig.transform.position;
        followCamera.transform.rotation = driverCameraRig.transform.rotation;
    }

    void SetEgoVisible(bool visible)
    {
        if (ego == null) return;
        Renderer[] renderers = ego.GetComponentsInChildren<Renderer>();
        foreach (Renderer renderer in renderers)
        {
            if (renderer != null && renderer.enabled != visible) renderer.enabled = visible;
        }
    }

    void EnsureStableOverheadFrame(DilSimState state, Vector3 egoPos)
    {
        if (overheadFrameReady) return;

        Bounds bounds = new Bounds(egoPos, new Vector3(20.0f, 1.0f, 12.0f));
        if (state.vehicles != null)
        {
            foreach (DilVehicleState v in state.vehicles)
            {
                bounds.Encapsulate(ToUnityPosition(v.x, v.y));
            }
        }

        float aspect = Mathf.Max(0.1f, followCamera == null ? 1.6f : followCamera.aspect);
        float halfLongitudinal = Mathf.Max(35.0f, bounds.size.x * 0.5f);
        float halfLateral = Mathf.Max(10.0f, bounds.size.z * 0.5f);
        overheadFrameCenter = bounds.center + Vector3.right * 20.0f;
        overheadFrameCenter.y = 0.0f;
        overheadFrameSize = Mathf.Clamp(Mathf.Max(halfLateral, halfLongitudinal / aspect) * 1.35f, 22.0f, 58.0f);
        overheadFrameReady = true;
    }

    Bounds CurrentTrafficBounds(Vector3 egoPos)
    {
        Bounds bounds = new Bounds(egoPos, new Vector3(12.0f, 1.0f, 8.0f));
        foreach (var pair in vehicles)
        {
            if (pair.Value == null) continue;
            bounds.Encapsulate(pair.Value.transform.position);
        }
        return bounds;
    }

    void UpdateHud(DilSimState state)
    {
        if (hudText == null) return;
        hudText.text =
            $"t = {state.time_s:F1} s\n" +
            $"v = {state.ego.speed:F1} m/s\n" +
            $"Front = {state.risk.front_distance_m:F1} m\n" +
            $"Collision = {state.safety.collision}";
    }

    void EnsureHudCanvas()
    {
        Canvas canvas = hudText == null ? null : hudText.GetComponentInParent<Canvas>();
        if (canvas == null)
        {
            canvas = GameObject.FindObjectOfType<Canvas>();
        }
        if (canvas == null)
        {
            GameObject canvasObject = new GameObject("HUD Canvas");
            canvas = canvasObject.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            CanvasScaler scaler = canvasObject.AddComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1920.0f, 1080.0f);
            canvasObject.AddComponent<GraphicRaycaster>();
        }
        if (hudText == null)
        {
            GameObject textObject = new GameObject("HUD Text");
            textObject.transform.SetParent(canvas.transform, false);
            Text text = textObject.AddComponent<Text>();
            text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            if (text.font == null)
            {
                text.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
            }
            text.fontSize = 22;
            text.color = Color.white;
            text.alignment = TextAnchor.UpperLeft;
            RectTransform rect = text.GetComponent<RectTransform>();
            rect.anchorMin = new Vector2(0.0f, 1.0f);
            rect.anchorMax = new Vector2(0.0f, 1.0f);
            rect.pivot = new Vector2(0.0f, 1.0f);
            rect.anchoredPosition = new Vector2(24.0f, -24.0f);
            rect.sizeDelta = new Vector2(360.0f, 140.0f);
            hudText = text;
        }
    }

    void UpdateRiskWarning(DilSimState state)
    {
        if (riskWarningText == null) return;
        bool driverActivelyTurning = DriverActivelyTurning(state);
        bool conflict = driverActivelyTurning && HasHumanMachineIntentConflict(state);
        bool humanIntentRisk = HumanIntentHasCollisionRisk(state);
        bool triggerWarning = driverActivelyTurning && conflict && humanIntentRisk;
        if (triggerWarning)
        {
            warningVisibleUntil = Time.time + 0.55f;
        }
        bool showWarning = Time.time <= warningVisibleUntil;
        if (!showWarning)
        {
            riskWarningText.enabled = false;
            return;
        }

        float blink = Mathf.Sin(Time.time * Mathf.PI * 2.0f * warningBlinkFrequencyHz);
        riskWarningText.enabled = blink > -0.25f;
        riskWarningText.text = "HUMAN-MACHINE CONFLICT";
    }

    void EnsureRiskWarningText()
    {
        if (riskWarningText != null) return;
        Canvas canvas = hudText == null ? null : hudText.GetComponentInParent<Canvas>();
        if (canvas == null) return;

        GameObject textObject = new GameObject("Risk Warning Text");
        textObject.transform.SetParent(canvas.transform, false);
        Text text = textObject.AddComponent<Text>();
        text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        if (text.font == null)
        {
            text.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
        }
        text.fontSize = 42;
        text.fontStyle = FontStyle.Bold;
        text.color = new Color(1.0f, 0.14f, 0.08f, 0.96f);
        text.alignment = TextAnchor.MiddleCenter;
        text.text = "HUMAN-MACHINE CONFLICT";
        text.enabled = false;

        RectTransform rect = text.GetComponent<RectTransform>();
        rect.anchorMin = new Vector2(0.5f, 1.0f);
        rect.anchorMax = new Vector2(0.5f, 1.0f);
        rect.pivot = new Vector2(0.5f, 1.0f);
        rect.anchoredPosition = new Vector2(0.0f, -188.0f);
        rect.sizeDelta = new Vector2(860.0f, 84.0f);

        Outline outline = textObject.AddComponent<Outline>();
        outline.effectColor = new Color(0.0f, 0.0f, 0.0f, 0.85f);
        outline.effectDistance = new Vector2(2.0f, -2.0f);
        riskWarningText = text;
    }

    void EnsureRearViewMirrors()
    {
        if (!showRearViewMirrors || followCamera == null) return;
        if (leftMirrorCamera != null || rightMirrorCamera != null || centerMirrorCamera != null) return;
        leftMirrorTexture = new RenderTexture(mirrorTextureWidth, mirrorTextureHeight, 16);
        rightMirrorTexture = new RenderTexture(mirrorTextureWidth, mirrorTextureHeight, 16);
        centerMirrorTexture = new RenderTexture(centerMirrorTextureWidth, centerMirrorTextureHeight, 16);
        leftMirrorCamera = CreateMirrorCamera("Left_Rear_View_Camera", leftMirrorTexture);
        rightMirrorCamera = CreateMirrorCamera("Right_Rear_View_Camera", rightMirrorTexture);
        centerMirrorCamera = CreateMirrorCamera("Cabin_Rear_View_Camera", centerMirrorTexture);
        Canvas canvas = hudText == null ? null : hudText.GetComponentInParent<Canvas>();
        if (canvas == null) return;
        leftMirrorImage = CreateMirrorImage(
            canvas.transform,
            "Left Side Mirror",
            leftMirrorTexture,
            new Vector2(30.0f, 34.0f),
            new Vector2(mirrorTextureWidth, mirrorTextureHeight),
            new Vector2(0.0f, 0.36f),
            new Vector2(0.0f, 0.36f),
            new Vector2(0.0f, 0.5f),
            "LEFT MIRROR"
        );
        rightMirrorImage = CreateMirrorImage(
            canvas.transform,
            "Right Side Mirror",
            rightMirrorTexture,
            new Vector2(-30.0f, 34.0f),
            new Vector2(mirrorTextureWidth, mirrorTextureHeight),
            new Vector2(1.0f, 0.36f),
            new Vector2(1.0f, 0.36f),
            new Vector2(1.0f, 0.5f),
            "RIGHT MIRROR"
        );
        centerMirrorImage = CreateCenterMirrorImage(canvas.transform, "Cabin Rear View", centerMirrorTexture);
    }

    void EnsureCockpitOverlay()
    {
        if (!showCockpitOverlay || steeringWheelOverlay != null) return;
        Canvas canvas = hudText == null ? null : hudText.GetComponentInParent<Canvas>();
        if (canvas == null) return;

        GameObject dashObject = new GameObject("Driver Cockpit Dashboard");
        dashObject.transform.SetParent(canvas.transform, false);
        Image dash = dashObject.AddComponent<Image>();
        dash.color = new Color(0.018f, 0.020f, 0.024f, 0.72f);
        RectTransform dashRect = dash.GetComponent<RectTransform>();
        dashRect.anchorMin = new Vector2(0.0f, 0.0f);
        dashRect.anchorMax = new Vector2(1.0f, 0.0f);
        dashRect.pivot = new Vector2(0.5f, 0.0f);
        dashRect.anchoredPosition = Vector2.zero;
        dashRect.sizeDelta = new Vector2(0.0f, 180.0f);

        GameObject wheelObject = new GameObject("Driver Steering Wheel");
        wheelObject.transform.SetParent(canvas.transform, false);
        SteeringWheelGraphic wheel = wheelObject.AddComponent<SteeringWheelGraphic>();
        wheel.color = new Color(0.01f, 0.012f, 0.015f, 0.98f);
        wheel.rimColor = new Color(0.86f, 0.90f, 0.96f, 0.98f);
        RectTransform wheelRect = wheel.GetComponent<RectTransform>();
        wheelRect.anchorMin = new Vector2(0.5f, 0.0f);
        wheelRect.anchorMax = new Vector2(0.5f, 0.0f);
        wheelRect.pivot = new Vector2(0.5f, 0.5f);
        wheelRect.anchoredPosition = new Vector2(0.0f, 98.0f);
        wheelRect.sizeDelta = new Vector2(290.0f, 290.0f);
        steeringWheelOverlay = wheelRect;

        dashObject.transform.SetAsLastSibling();
        wheelObject.transform.SetAsLastSibling();
        if (leftMirrorImage != null) leftMirrorImage.transform.parent.SetAsLastSibling();
        if (rightMirrorImage != null) rightMirrorImage.transform.parent.SetAsLastSibling();
        if (centerMirrorImage != null) centerMirrorImage.transform.parent.SetAsLastSibling();
        if (riskWarningText != null) riskWarningText.transform.SetAsLastSibling();
    }

    void UpdateCockpitOverlay(DilSimState state)
    {
        if (steeringWheelOverlay == null || state == null) return;
        float steer = 0.0f;
        if (state.driver_input != null)
        {
            steer = state.driver_input.steer;
            if (Mathf.Abs(steer) < 0.01f)
            {
                steer = state.driver_input.delta_rad / 0.20f;
            }
        }
        steeringWheelOverlay.localRotation = Quaternion.Euler(0.0f, 0.0f, -Mathf.Clamp(steer, -1.0f, 1.0f) * 120.0f);
        steeringWheelOverlay.SetAsLastSibling();
        BringDriverUiToFront();
    }

    void EnsureCockpitModel()
    {
        if (!showCockpitModel || followCamera == null || cockpitRoot != null) return;

        cockpitRoot = new GameObject("Driver Cockpit Model");
        cockpitRoot.transform.SetParent(followCamera.transform, false);
        cockpitRoot.transform.localPosition = Vector3.zero;
        cockpitRoot.transform.localRotation = Quaternion.identity;
        cockpitRoot.transform.localScale = Vector3.one;

        CreateCockpitCube("Dashboard", new Vector3(0.0f, -0.54f, 0.82f), new Vector3(2.35f, 0.20f, 0.42f), cockpitMaterial, Quaternion.identity);
        CreateCockpitCube("Instrument Cluster", new Vector3(0.0f, -0.36f, 0.58f), new Vector3(0.72f, 0.10f, 0.18f), cockpitTrimMaterial, Quaternion.identity);
        CreateCockpitCube("Windshield Lower Frame", new Vector3(0.0f, -0.18f, 0.94f), new Vector3(2.35f, 0.055f, 0.08f), cockpitTrimMaterial, Quaternion.identity);
        CreateCockpitCube("Windshield Upper Frame", new Vector3(0.0f, 0.58f, 1.05f), new Vector3(2.60f, 0.060f, 0.08f), cockpitTrimMaterial, Quaternion.identity);
        CreateCockpitCube("Left A Pillar", new Vector3(-1.10f, 0.20f, 0.97f), new Vector3(0.08f, 1.20f, 0.08f), cockpitTrimMaterial, Quaternion.Euler(0.0f, 0.0f, -13.0f));
        CreateCockpitCube("Right A Pillar", new Vector3(1.10f, 0.20f, 0.97f), new Vector3(0.08f, 1.20f, 0.08f), cockpitTrimMaterial, Quaternion.Euler(0.0f, 0.0f, 13.0f));
        CreateCockpitCube("Left Door Top", new Vector3(-1.18f, -0.34f, 0.78f), new Vector3(0.08f, 0.16f, 0.80f), cockpitMaterial, Quaternion.identity);
        CreateCockpitCube("Right Door Top", new Vector3(1.18f, -0.34f, 0.78f), new Vector3(0.08f, 0.16f, 0.80f), cockpitMaterial, Quaternion.identity);

        GameObject wheelRootObject = new GameObject("Cockpit Steering Wheel");
        wheelRootObject.transform.SetParent(cockpitRoot.transform, false);
        wheelRootObject.transform.localPosition = new Vector3(0.0f, -0.33f, 0.46f);
        wheelRootObject.transform.localRotation = Quaternion.Euler(0.0f, 0.0f, 0.0f);
        wheelRootObject.transform.localScale = Vector3.one;
        cockpitWheelRoot = wheelRootObject.transform;
        CreateCockpitWheelRing(cockpitWheelRoot, 0.23f, 48, 0.030f);
        CreateCockpitWheelSpoke(cockpitWheelRoot, new Vector3(0.0f, 0.0f, 0.0f), new Vector3(0.0f, 0.18f, 0.0f), 0.020f);
        CreateCockpitWheelSpoke(cockpitWheelRoot, new Vector3(0.0f, 0.0f, 0.0f), new Vector3(-0.17f, -0.13f, 0.0f), 0.020f);
        CreateCockpitWheelSpoke(cockpitWheelRoot, new Vector3(0.0f, 0.0f, 0.0f), new Vector3(0.17f, -0.13f, 0.0f), 0.020f);
        CreateCockpitCube("Steering Hub", new Vector3(0.0f, -0.33f, 0.455f), new Vector3(0.18f, 0.18f, 0.035f), cockpitTrimMaterial, Quaternion.identity);
    }

    GameObject CreateCockpitCube(string name, Vector3 localPosition, Vector3 localScale, Material material, Quaternion localRotation)
    {
        GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        cube.name = name;
        cube.transform.SetParent(cockpitRoot.transform, false);
        cube.transform.localPosition = localPosition;
        cube.transform.localRotation = localRotation;
        cube.transform.localScale = localScale;
        Renderer renderer = cube.GetComponent<Renderer>();
        if (renderer != null && material != null)
        {
            renderer.sharedMaterial = material;
        }
        return cube;
    }

    void CreateCockpitWheelRing(Transform parent, float radius, int segments, float width)
    {
        GameObject ringObject = new GameObject("Steering Wheel Rim");
        ringObject.transform.SetParent(parent, false);
        LineRenderer ring = ringObject.AddComponent<LineRenderer>();
        ring.useWorldSpace = false;
        ring.loop = true;
        ring.positionCount = segments;
        ring.startWidth = width;
        ring.endWidth = width;
        ring.material = cockpitTrimMaterial;
        for (int i = 0; i < segments; i++)
        {
            float angle = (Mathf.PI * 2.0f * i) / segments;
            ring.SetPosition(i, new Vector3(Mathf.Cos(angle) * radius, Mathf.Sin(angle) * radius, 0.0f));
        }
    }

    void CreateCockpitWheelSpoke(Transform parent, Vector3 from, Vector3 to, float width)
    {
        GameObject spokeObject = new GameObject("Steering Wheel Spoke");
        spokeObject.transform.SetParent(parent, false);
        LineRenderer spoke = spokeObject.AddComponent<LineRenderer>();
        spoke.useWorldSpace = false;
        spoke.positionCount = 2;
        spoke.startWidth = width;
        spoke.endWidth = width;
        spoke.material = cockpitTrimMaterial;
        spoke.SetPosition(0, from);
        spoke.SetPosition(1, to);
    }

    void UpdateCockpitModel(DilSimState state)
    {
        if (cockpitWheelRoot == null || state == null) return;
        float steer = 0.0f;
        if (state.driver_input != null)
        {
            steer = state.driver_input.steer;
            if (Mathf.Abs(steer) < 0.01f)
            {
                steer = state.driver_input.delta_rad / 0.20f;
            }
        }
        cockpitWheelRoot.localRotation = Quaternion.Euler(0.0f, 0.0f, -Mathf.Clamp(steer, -1.0f, 1.0f) * 95.0f);
    }

    void BringDriverUiToFront()
    {
        if (leftMirrorImage != null && leftMirrorImage.transform.parent != null) leftMirrorImage.transform.parent.SetAsLastSibling();
        if (rightMirrorImage != null && rightMirrorImage.transform.parent != null) rightMirrorImage.transform.parent.SetAsLastSibling();
        if (centerMirrorImage != null && centerMirrorImage.transform.parent != null) centerMirrorImage.transform.parent.SetAsLastSibling();
        if (steeringWheelOverlay != null) steeringWheelOverlay.SetAsLastSibling();
        if (riskWarningText != null) riskWarningText.transform.SetAsLastSibling();
    }

    Camera CreateMirrorCamera(string cameraName, RenderTexture texture)
    {
        GameObject cameraObject = new GameObject(cameraName);
        Camera camera = cameraObject.AddComponent<Camera>();
        camera.clearFlags = CameraClearFlags.Skybox;
        camera.fieldOfView = mirrorCameraFov;
        camera.nearClipPlane = 0.05f;
        camera.farClipPlane = 350.0f;
        camera.targetTexture = texture;
        camera.depth = -10;
        return camera;
    }

    RawImage CreateMirrorImage(
        Transform parent,
        string imageName,
        RenderTexture texture,
        Vector2 anchoredPosition,
        Vector2 size,
        Vector2 anchorMin,
        Vector2 anchorMax,
        Vector2 pivot,
        string label
    )
    {
        GameObject frame = new GameObject(imageName + " Frame");
        frame.transform.SetParent(parent, false);
        Image frameImage = frame.AddComponent<Image>();
        frameImage.color = new Color(0.02f, 0.025f, 0.03f, 0.82f);
        RectTransform frameRect = frame.GetComponent<RectTransform>();
        frameRect.anchorMin = anchorMin;
        frameRect.anchorMax = anchorMax;
        frameRect.pivot = pivot;
        frameRect.anchoredPosition = anchoredPosition;
        frameRect.sizeDelta = new Vector2(size.x + 22.0f, size.y + 36.0f);

        GameObject imageObject = new GameObject(imageName);
        imageObject.transform.SetParent(frame.transform, false);
        RawImage image = imageObject.AddComponent<RawImage>();
        image.texture = texture;
        image.color = new Color(1.0f, 1.0f, 1.0f, 0.92f);
        RectTransform rect = image.GetComponent<RectTransform>();
        rect.anchorMin = new Vector2(0.5f, 0.5f);
        rect.anchorMax = new Vector2(0.5f, 0.5f);
        rect.pivot = new Vector2(0.5f, 0.5f);
        rect.anchoredPosition = new Vector2(0.0f, -7.0f);
        rect.sizeDelta = size;

        Outline outline = frame.AddComponent<Outline>();
        outline.effectColor = new Color(0.95f, 0.98f, 1.0f, 0.96f);
        outline.effectDistance = new Vector2(2.0f, -2.0f);
        AddMirrorLabel(frame.transform, label, new Vector2(0.0f, size.y * 0.5f + 5.0f), new Vector2(size.x, 24.0f));
        frame.transform.SetAsLastSibling();
        return image;
    }

    RawImage CreateCenterMirrorImage(Transform parent, string imageName, RenderTexture texture)
    {
        GameObject frame = new GameObject(imageName + " Frame");
        frame.transform.SetParent(parent, false);
        Image frameImage = frame.AddComponent<Image>();
        frameImage.color = new Color(0.02f, 0.025f, 0.03f, 0.86f);
        RectTransform frameRect = frame.GetComponent<RectTransform>();
        frameRect.anchorMin = new Vector2(0.5f, 1.0f);
        frameRect.anchorMax = new Vector2(0.5f, 1.0f);
        frameRect.pivot = new Vector2(0.5f, 1.0f);
        frameRect.anchoredPosition = new Vector2(0.0f, -18.0f);
        frameRect.sizeDelta = new Vector2(centerMirrorTextureWidth + 28.0f, centerMirrorTextureHeight + 22.0f);

        GameObject imageObject = new GameObject(imageName);
        imageObject.transform.SetParent(frame.transform, false);
        RawImage image = imageObject.AddComponent<RawImage>();
        image.texture = texture;
        image.color = new Color(1.0f, 1.0f, 1.0f, 0.94f);
        RectTransform rect = image.GetComponent<RectTransform>();
        rect.anchorMin = new Vector2(0.5f, 0.5f);
        rect.anchorMax = new Vector2(0.5f, 0.5f);
        rect.pivot = new Vector2(0.5f, 0.5f);
        rect.anchoredPosition = Vector2.zero;
        rect.sizeDelta = new Vector2(centerMirrorTextureWidth, centerMirrorTextureHeight);

        Outline outline = frame.AddComponent<Outline>();
        outline.effectColor = new Color(1.0f, 1.0f, 1.0f, 0.82f);
        outline.effectDistance = new Vector2(2.0f, -2.0f);
        AddMirrorLabel(frame.transform, "REAR VIEW", new Vector2(0.0f, centerMirrorTextureHeight * 0.5f + 3.0f), new Vector2(centerMirrorTextureWidth, 24.0f));
        frame.transform.SetAsLastSibling();
        return image;
    }

    void AddMirrorLabel(Transform parent, string label, Vector2 anchoredPosition, Vector2 size)
    {
        GameObject labelObject = new GameObject(label);
        labelObject.transform.SetParent(parent, false);
        Text text = labelObject.AddComponent<Text>();
        text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        if (text.font == null)
        {
            text.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
        }
        text.fontSize = 15;
        text.fontStyle = FontStyle.Bold;
        text.alignment = TextAnchor.MiddleCenter;
        text.color = new Color(1.0f, 1.0f, 1.0f, 0.95f);
        text.text = label;
        RectTransform rect = text.GetComponent<RectTransform>();
        rect.anchorMin = new Vector2(0.5f, 0.5f);
        rect.anchorMax = new Vector2(0.5f, 0.5f);
        rect.pivot = new Vector2(0.5f, 0.5f);
        rect.anchoredPosition = anchoredPosition;
        rect.sizeDelta = size;
        Outline outline = labelObject.AddComponent<Outline>();
        outline.effectColor = new Color(0.0f, 0.0f, 0.0f, 0.9f);
        outline.effectDistance = new Vector2(1.0f, -1.0f);
    }

    void UpdateRearViewMirrors(DilSimState state)
    {
        if (!showRearViewMirrors || leftMirrorCamera == null || rightMirrorCamera == null || ego == null) return;
        Vector3 egoPos = ego.transform.position;
        Quaternion baseRotation = ego.transform.rotation;
        Vector3 forward = baseRotation * Vector3.right;
        Vector3 right = baseRotation * Vector3.back;
        PositionMirrorCamera(leftMirrorCamera, egoPos, forward, right, -mirrorCameraLateralOffset, -mirrorCameraYawDeg, mirrorCameraHeight, mirrorCameraBackOffset);
        PositionMirrorCamera(rightMirrorCamera, egoPos, forward, right, mirrorCameraLateralOffset, mirrorCameraYawDeg, mirrorCameraHeight, mirrorCameraBackOffset);
        if (centerMirrorCamera != null)
        {
            PositionMirrorCamera(centerMirrorCamera, egoPos, forward, right, 0.0f, centerMirrorYawDeg, mirrorCameraHeight + 0.18f, mirrorCameraBackOffset + 0.10f);
        }
    }

    void PositionMirrorCamera(Camera mirrorCamera, Vector3 egoPos, Vector3 forward, Vector3 right, float lateralOffset, float yawDeg, float height, float backOffset)
    {
        Vector3 pos =
            egoPos
            + forward * backOffset
            + right * lateralOffset
            + Vector3.up * height;
        Quaternion lookBack = Quaternion.LookRotation(-forward, Vector3.up);
        Quaternion yawOffset = Quaternion.AngleAxis(yawDeg, Vector3.up);
        mirrorCamera.transform.position = pos;
        mirrorCamera.transform.rotation = lookBack * yawOffset;
        mirrorCamera.fieldOfView = mirrorCameraFov;
    }

    bool HasHumanMachineConflict(DilSimState state)
    {
        if (state == null || state.driver_input == null || state.ego == null) return false;
        float driverSteer = state.driver_input.delta_rad;
        float sharedSteer = state.ego.steer;
        return Mathf.Abs(driverSteer - sharedSteer) >= conflictWarningThresholdRad;
    }

    bool DriverActivelyTurning(DilSimState state)
    {
        if (state == null || state.driver_input == null) return false;
        float steerNorm = Mathf.Abs(state.driver_input.steer);
        float steerRad = Mathf.Abs(state.driver_input.delta_rad);
        return steerNorm >= 0.55f || steerRad >= 0.14f;
    }

    bool HasHumanMachineIntentConflict(DilSimState state)
    {
        if (state == null || state.driver_input == null || !DriverActivelyTurning(state)) return false;
        float humanDirection = TrajectoryLateralDirection(state.intention == null ? null : state.intention.human);
        float driverSteer = state.driver_input.delta_rad;
        if (Mathf.Abs(driverSteer) < 0.01f)
        {
            driverSteer = state.driver_input.steer * 0.20f;
        }
        float driverDirection = Mathf.Sign(driverSteer);
        if (Mathf.Abs(humanDirection) < 0.15f || Mathf.Sign(humanDirection) != driverDirection)
        {
            humanDirection = driverDirection * Mathf.Min(1.0f, Mathf.Abs(driverSteer) / Mathf.Max(0.16f, conflictWarningThresholdRad));
        }

        float machineDirection = TrajectoryLateralDirection(state.intention == null ? null : state.intention.machine);
        if (Mathf.Abs(machineDirection) < 0.15f && state.ego != null && Mathf.Abs(state.ego.steer) >= conflictWarningThresholdRad)
        {
            machineDirection = Mathf.Sign(state.ego.steer) * Mathf.Min(1.0f, Mathf.Abs(state.ego.steer) / Mathf.Max(conflictWarningThresholdRad, 0.001f));
        }

        bool bothTurningOpposite =
            Mathf.Abs(humanDirection) >= 0.85f &&
            Mathf.Abs(machineDirection) >= 0.65f &&
            Mathf.Sign(humanDirection) != Mathf.Sign(machineDirection);
        return bothTurningOpposite;
    }

    float TrajectoryLateralDirection(DilPointState[] trajectory)
    {
        if (trajectory == null || trajectory.Length < 2) return 0.0f;
        float startY = trajectory[0].y;
        float endY = trajectory[trajectory.Length - 1].y;
        float dy = endY - startY;
        if (Mathf.Abs(dy) < 0.25f) return 0.0f;
        return Mathf.Sign(dy);
    }

    bool HumanIntentHasCollisionRisk(DilSimState state)
    {
        if (state == null) return false;
        DilPointState[] humanTrajectory = state.intention == null ? null : state.intention.human;
        if (humanTrajectory == null || humanTrajectory.Length < 2)
        {
            bool closeFront = state.risk != null && state.risk.front_distance_m > 0.0f && state.risk.front_distance_m < 34.0f;
            bool lowTtc = state.risk != null && state.risk.ttc_s > 0.0f && state.risk.ttc_s < 4.0f;
            return closeFront || lowTtc;
        }
        if (state.vehicles == null) return false;

        float egoLength = state.ego == null ? 4.6f : Mathf.Max(4.2f, state.ego.length);
        float egoWidth = state.ego == null ? 1.8f : Mathf.Max(1.6f, state.ego.width);
        foreach (DilPointState p in humanTrajectory)
        {
            foreach (DilVehicleState v in state.vehicles)
            {
                float longLimit = 0.5f * (egoLength + Mathf.Max(4.0f, v.length)) + 1.2f;
                float latLimit = 0.5f * (egoWidth + Mathf.Max(1.6f, v.width)) + 0.45f;
                if (Mathf.Abs(p.x - v.x) <= longLimit && Mathf.Abs(p.y - v.y) <= latLimit)
                {
                    return true;
                }
            }
        }
        return false;
    }

    string CameraModeName()
    {
        if (cameraMode == 0) return "driver";
        if (cameraMode == 2) return "overhead";
        return "chase";
    }

    int CountObjectsNamed(string objectName)
    {
        int count = 0;
        foreach (GameObject obj in GameObject.FindObjectsOfType<GameObject>())
        {
            if (obj.name == objectName && obj.activeInHierarchy) count += 1;
        }
        return count;
    }
}
