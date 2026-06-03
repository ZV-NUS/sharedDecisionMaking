using UnityEngine;
using UnityEngine.UI;

public class MirrorFrameGraphic : Graphic
{
    public float topInset = 34.0f;

    protected override void OnPopulateMesh(VertexHelper vh)
    {
        vh.Clear();
        Rect r = rectTransform.rect;
        float inset = Mathf.Clamp(topInset, 0.0f, r.width * 0.35f);
        Vector2 p0 = new Vector2(r.xMin + inset, r.yMax);
        Vector2 p1 = new Vector2(r.xMax - inset, r.yMax);
        Vector2 p2 = new Vector2(r.xMax, r.yMin);
        Vector2 p3 = new Vector2(r.xMin, r.yMin);

        UIVertex v = UIVertex.simpleVert;
        v.color = color;
        v.position = p0;
        vh.AddVert(v);
        v.position = p1;
        vh.AddVert(v);
        v.position = p2;
        vh.AddVert(v);
        v.position = p3;
        vh.AddVert(v);
        vh.AddTriangle(0, 1, 2);
        vh.AddTriangle(2, 3, 0);
    }
}
