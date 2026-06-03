using UnityEngine;
using UnityEngine.UI;

public class SteeringWheelGraphic : Graphic
{
    public Color rimColor = new Color(0.92f, 0.94f, 0.96f, 0.92f);
    public int segments = 72;
    public float outerRadius = 0.48f;
    public float innerRadius = 0.36f;
    public float hubRadius = 0.11f;

    protected override void OnPopulateMesh(VertexHelper vh)
    {
        vh.Clear();
        Rect rect = rectTransform.rect;
        Vector2 center = rect.center;
        float radius = Mathf.Min(rect.width, rect.height) * 0.5f;
        int count = Mathf.Clamp(segments, 24, 160);

        AddRing(vh, center, radius * outerRadius, radius * innerRadius, color);
        AddHub(vh, center, radius * hubRadius, rimColor, count);
        AddSpoke(vh, center, radius * hubRadius * 0.7f, radius * innerRadius, 90.0f, radius * 0.035f, rimColor);
        AddSpoke(vh, center, radius * hubRadius * 0.7f, radius * innerRadius, 220.0f, radius * 0.035f, rimColor);
        AddSpoke(vh, center, radius * hubRadius * 0.7f, radius * innerRadius, 320.0f, radius * 0.035f, rimColor);
    }

    void AddRing(VertexHelper vh, Vector2 center, float outer, float inner, Color c)
    {
        int count = Mathf.Clamp(segments, 24, 160);
        for (int i = 0; i < count; ++i)
        {
            float a0 = Mathf.PI * 2.0f * i / count;
            float a1 = Mathf.PI * 2.0f * (i + 1) / count;
            Vector2 p0 = center + new Vector2(Mathf.Cos(a0), Mathf.Sin(a0)) * outer;
            Vector2 p1 = center + new Vector2(Mathf.Cos(a1), Mathf.Sin(a1)) * outer;
            Vector2 p2 = center + new Vector2(Mathf.Cos(a1), Mathf.Sin(a1)) * inner;
            Vector2 p3 = center + new Vector2(Mathf.Cos(a0), Mathf.Sin(a0)) * inner;
            AddQuad(vh, p0, p1, p2, p3, c);
        }
    }

    void AddHub(VertexHelper vh, Vector2 center, float radius, Color c, int count)
    {
        for (int i = 0; i < count; ++i)
        {
            float a0 = Mathf.PI * 2.0f * i / count;
            float a1 = Mathf.PI * 2.0f * (i + 1) / count;
            Vector2 p0 = center;
            Vector2 p1 = center + new Vector2(Mathf.Cos(a0), Mathf.Sin(a0)) * radius;
            Vector2 p2 = center + new Vector2(Mathf.Cos(a1), Mathf.Sin(a1)) * radius;
            AddTriangle(vh, p0, p1, p2, c);
        }
    }

    void AddSpoke(VertexHelper vh, Vector2 center, float inner, float outer, float deg, float halfWidth, Color c)
    {
        float angle = deg * Mathf.Deg2Rad;
        Vector2 dir = new Vector2(Mathf.Cos(angle), Mathf.Sin(angle));
        Vector2 normal = new Vector2(-dir.y, dir.x);
        Vector2 p0 = center + dir * inner + normal * halfWidth;
        Vector2 p1 = center + dir * outer + normal * halfWidth;
        Vector2 p2 = center + dir * outer - normal * halfWidth;
        Vector2 p3 = center + dir * inner - normal * halfWidth;
        AddQuad(vh, p0, p1, p2, p3, c);
    }

    void AddQuad(VertexHelper vh, Vector2 p0, Vector2 p1, Vector2 p2, Vector2 p3, Color c)
    {
        int start = vh.currentVertCount;
        AddVert(vh, p0, c);
        AddVert(vh, p1, c);
        AddVert(vh, p2, c);
        AddVert(vh, p3, c);
        vh.AddTriangle(start, start + 1, start + 2);
        vh.AddTriangle(start + 2, start + 3, start);
    }

    void AddTriangle(VertexHelper vh, Vector2 p0, Vector2 p1, Vector2 p2, Color c)
    {
        int start = vh.currentVertCount;
        AddVert(vh, p0, c);
        AddVert(vh, p1, c);
        AddVert(vh, p2, c);
        vh.AddTriangle(start, start + 1, start + 2);
    }

    void AddVert(VertexHelper vh, Vector2 pos, Color c)
    {
        UIVertex v = UIVertex.simpleVert;
        v.color = c;
        v.position = pos;
        vh.AddVert(v);
    }
}
