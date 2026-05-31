using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class UnityDriverInputSender : MonoBehaviour
{
    public string pythonHost = "127.0.0.1";
    public int pythonInputPort = 50711;
    public float maxSteerRad = 0.075f;
    public float maxThrottleAccel = 4.2f;
    public float maxBrakeDecel = 7.0f;
    public float steerRiseRate = 0.45f;
    public float steerReturnRate = 1.05f;
    public float steerShapeExponent = 1.65f;
    public bool forceDampedKeyboardPreset = true;

    private UdpClient udp;
    private IPEndPoint endpoint;
    private float steer;

    void Start()
    {
        if (forceDampedKeyboardPreset)
        {
            maxSteerRad = Mathf.Min(maxSteerRad, 0.075f);
            steerRiseRate = Mathf.Min(steerRiseRate, 0.45f);
            steerReturnRate = Mathf.Min(steerReturnRate, 1.05f);
            steerShapeExponent = Mathf.Max(steerShapeExponent, 1.65f);
            maxThrottleAccel = Mathf.Max(maxThrottleAccel, 4.2f);
            maxBrakeDecel = Mathf.Max(maxBrakeDecel, 7.0f);
        }
        udp = new UdpClient();
        endpoint = new IPEndPoint(IPAddress.Parse(pythonHost), pythonInputPort);
        Debug.Log($"Unity driver input sender -> {pythonHost}:{pythonInputPort}");
    }

    void Update()
    {
        float targetSteer = 0.0f;
        if (Input.GetKey(KeyCode.LeftArrow) || Input.GetKey(KeyCode.A)) targetSteer -= 1.0f;
        if (Input.GetKey(KeyCode.RightArrow) || Input.GetKey(KeyCode.D)) targetSteer += 1.0f;

        float rate = Mathf.Abs(targetSteer) > 0.01f ? steerRiseRate : steerReturnRate;
        steer = Mathf.MoveTowards(steer, targetSteer, rate * Time.deltaTime);

        float throttle = (Input.GetKey(KeyCode.UpArrow) || Input.GetKey(KeyCode.W)) ? 1.0f : 0.0f;
        float brake = (Input.GetKey(KeyCode.DownArrow) || Input.GetKey(KeyCode.S)) ? 1.0f : 0.0f;
        if (Input.GetKey(KeyCode.PageUp)) throttle = 1.0f;
        if (Input.GetKey(KeyCode.PageDown)) brake = 1.0f;
        if (Input.GetKey(KeyCode.Space))
        {
            brake = 1.0f;
            throttle = 0.0f;
        }

        bool reset = Input.GetKeyDown(KeyCode.R);
        bool quit = Input.GetKeyDown(KeyCode.Q);
        float shapedSteer = Mathf.Sign(steer) * Mathf.Pow(Mathf.Abs(steer), Mathf.Max(1.0f, steerShapeExponent));
        float delta = maxSteerRad * shapedSteer;
        float accel = maxThrottleAccel * throttle - maxBrakeDecel * brake;

        string json =
            "{" +
            "\"source\":\"unity\"," +
            "\"steer\":" + F(shapedSteer) + "," +
            "\"throttle\":" + F(throttle) + "," +
            "\"brake\":" + F(brake) + "," +
            "\"delta_rad\":" + F(delta) + "," +
            "\"acceleration_mps2\":" + F(accel) + "," +
            "\"ready\":true," +
            "\"reset\":" + (reset ? "true" : "false") + "," +
            "\"quit\":" + (quit ? "true" : "false") +
            "}";

        byte[] data = Encoding.UTF8.GetBytes(json);
        udp.Send(data, data.Length, endpoint);
    }

    string F(float value)
    {
        return value.ToString("0.######", System.Globalization.CultureInfo.InvariantCulture);
    }

    void OnDestroy()
    {
        udp?.Close();
    }
}
