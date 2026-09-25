// Camera finish for the Built-In pipeline without the post-processing package: a four-level
// bloom so emissive signs and windows halo, ACES tone mapping of the HDR buffer, a light
// grade and a vignette.
Shader "Hidden/Parity/PostFx"
{
    Properties { _MainTex ("", 2D) = "black" {} }
    CGINCLUDE
    #include "UnityCG.cginc"
    sampler2D _MainTex, _Bloom;
    float4 _MainTex_TexelSize;
    float _Threshold, _BloomStrength, _Exposure, _Vignette, _Saturation, _Contrast;
    float4 _Tint;

    half3 Box4(float2 uv)
    {
        float4 d = _MainTex_TexelSize.xyxy * float4(-1, -1, 1, 1);
        return (tex2D(_MainTex, uv + d.xy).rgb + tex2D(_MainTex, uv + d.zy).rgb +
                tex2D(_MainTex, uv + d.xw).rgb + tex2D(_MainTex, uv + d.zw).rgb) * 0.25;
    }
    half4 Prefilter(v2f_img i) : SV_Target
    {
        half3 c = Box4(i.uv);
        half br = max(c.r, max(c.g, c.b));
        half soft = saturate(br - _Threshold + 0.5);
        soft = soft * soft * 0.5;
        half w = max(soft, br - _Threshold) / max(br, 1e-4);
        return half4(c * w, 1);
    }
    half4 Down(v2f_img i) : SV_Target { return half4(Box4(i.uv), 1); }
    half4 Up(v2f_img i) : SV_Target { return half4(Box4(i.uv) * 0.5, 1); }

    half3 Aces(half3 x)
    {
        return saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
    }
    half4 Composite(v2f_img i) : SV_Target
    {
        half3 c = tex2D(_MainTex, i.uv).rgb + tex2D(_Bloom, i.uv).rgb * _BloomStrength;
        c = Aces(c * _Exposure);
        half l = dot(c, half3(0.2126, 0.7152, 0.0722));
        c = lerp(l.xxx, c, _Saturation);
        c = saturate((c - 0.5) * _Contrast + 0.5) * _Tint.rgb;
        float2 q = i.uv - 0.5;
        c *= 1 - _Vignette * smoothstep(0.25, 0.85, dot(q, q) * 2.2);
        return half4(c, 1);
    }
    ENDCG
    SubShader
    {
        Cull Off ZWrite Off ZTest Always
        Pass { CGPROGRAM
               #pragma vertex vert_img
               #pragma fragment Prefilter
               ENDCG }
        Pass { CGPROGRAM
               #pragma vertex vert_img
               #pragma fragment Down
               ENDCG }
        Pass { Blend One One
               CGPROGRAM
               #pragma vertex vert_img
               #pragma fragment Up
               ENDCG }
        Pass { CGPROGRAM
               #pragma vertex vert_img
               #pragma fragment Composite
               ENDCG }
    }
    Fallback Off
}
