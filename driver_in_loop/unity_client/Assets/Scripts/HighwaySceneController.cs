using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

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
    public bool useImportedVehiclePrefabs = true;
    public bool tintImportedVehiclePrefabs = false;
    public Vector3 importedVehicleEulerOffset = new Vector3(0.0f, 90.0f, 0.0f);
    public bool showMachineIntention = true;
    public float intentionLineWidth = 0.16f;
    public Text hudText;

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
    public float driverCameraHeight = 2.2f;
    public float driverCameraForwardOffset = 1.2f;
    public float driverCameraLookAhead = 80.0f;
    public float driverCameraLookHeight = 1.25f;
    public bool hideEgoInDriverView = true;
    public float driverCameraSmoothTime = 0.0f;
    public float driverCameraRotationFollowRate = 90.0f;
    public bool lockDriverCameraToRoad = true;
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
    private LineRenderer machineIntentionLine;

    void Start()
    {
        if (udpClient == null) udpClient = GetComponent<DilUdpClient>();
        if (followCamera == null) followCamera = Camera.main;
        EnsureMaterials();
        ApplyValidationPreset();
        EnsureRuntimeRoot();
        HideLegacyGeneratedObjects();
        ClearRuntimeRootChildren();
        EnsureDriverCameraRig();
        ego = CreateVehicle("ego", egoMaterial, true);
        DeactivateDuplicateEgos();
    }

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
    }

    void LateUpdate()
    {
        if (renderCameraState != null)
        {
            UpdateCamera(renderCameraState);
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
        fixedRoadInWorld = true;
        useVisualSmoothing = false;
        visualFollowRate = 18.0f;
        smoothEgoForCamera = false;
        egoCameraFollowRate = 22.0f;
        cameraPositionSmoothTime = 0.45f;
        cameraRotationLerp = 0.08f;
        overheadSizeSmoothTime = 0.65f;
        stableOverheadCamera = true;
        driverCameraHeight = 1.45f;
        driverCameraForwardOffset = 1.2f;
        driverCameraLookAhead = 80.0f;
        driverCameraLookHeight = 1.25f;
        hideEgoInDriverView = true;
        driverCameraSmoothTime = 0.0f;
        driverCameraRotationFollowRate = 90.0f;
        lockDriverCameraToRoad = true;
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
        GameObject road = GameObject.CreatePrimitive(PrimitiveType.Cube);
        road.name = "Road";
        if (runtimeRoot != null) road.transform.SetParent(runtimeRoot.transform, true);
        road.GetComponent<Renderer>().material = roadMaterial;
        road.transform.localScale = new Vector3(roadLengthAhead + roadLengthBehind, 0.03f, width);
        road.transform.position = new Vector3(0.0f, 0.0f, roadCenterZ);
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
        line.transform.position = new Vector3(0.0f, 0.045f, -mark * positionScale);
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
            dash.transform.position = new Vector3(x + segLength * 0.5f, 0.045f, -mark * positionScale);
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
        int egoObjectCount = CountObjectsNamed("ego");
        int runtimeChildCount = runtimeRoot == null ? 0 : runtimeRoot.transform.childCount;
        string caseLabel = state.paper_case_id > 0
            ? $"Paper Case {state.paper_case_id} / rollout {state.case_id}"
            : $"Rollout Case {state.case_id}";
        hudText.text =
            $"{caseLabel}\n" +
            $"mode = {state.mode}\n" +
            $"t = {state.time_s:F1} s\n" +
            $"v = {state.ego.speed:F1} m/s\n" +
            $"camera = {CameraModeName()} (C)\n" +
            $"render interp = {udpClient.useRenderInterpolation} / {udpClient.renderDelaySeconds:F2}s\n" +
            $"vehicles = {(state.vehicles == null ? 0 : state.vehicles.Length)}\n" +
            $"ego objects = {egoObjectCount} / runtime children = {runtimeChildCount}\n" +
            $"lambda_RL = {state.authority.rl:F2}\n" +
            $"TTC = {state.risk.ttc_s:F1} s\n" +
            $"Front = {state.risk.front_distance_m:F1} m\n" +
            $"Collision = {state.safety.collision}";
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
