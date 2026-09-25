// One ground quad per agent: a soft contact shadow that grounds the figure at every LOD, and
// a glowing ring whose strength is the agent's accumulated divergence. Under the baseline the
// rings build up and never go out; under PARITY they flare and are cleared by restoration.
Shader "Parity/CrowdDecal"
{
    Properties
    {
        _BaseColor ("Ring Colour", Color) = (1,0.3,0.1,1)
        _Heat ("Ring", Float) = 0
        _Blob ("Contact Shadow", Float) = 0.42
    }
    SubShader
    {
        Tags { "Queue"="Transparent-50" "RenderType"="Transparent" "IgnoreProjector"="True" }
        Blend SrcAlpha OneMinusSrcAlpha
        ZWrite Off
        Cull Off
        Offset -1, -1
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing
            #pragma multi_compile_fog
            #include "UnityCG.cginc"

            float _Blob;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };
            struct v2f
            {
                float4 pos : SV_POSITION;
                float2 uv : TEXCOORD0;
                half4 col : COLOR;
                UNITY_FOG_COORDS(1)
            };

            UNITY_INSTANCING_BUFFER_START(Props)
                UNITY_DEFINE_INSTANCED_PROP(float4, _BaseColor)
                UNITY_DEFINE_INSTANCED_PROP(float, _Heat)
            UNITY_INSTANCING_BUFFER_END(Props)

            v2f vert (appdata v)
            {
                v2f o;
                UNITY_SETUP_INSTANCE_ID(v);
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv * 2 - 1;
                o.col = UNITY_ACCESS_INSTANCED_PROP(Props, _BaseColor);
                o.col.a = UNITY_ACCESS_INSTANCED_PROP(Props, _Heat);
                UNITY_TRANSFER_FOG(o, o.pos);
                return o;
            }

            half4 frag (v2f i) : SV_Target
            {
                float r = length(i.uv);
                float blob = (1 - smoothstep(0.0, 0.62, r)) * _Blob;
                float ring = smoothstep(0.58, 0.66, r) * (1 - smoothstep(0.74, 0.86, r)) * i.col.a;
                float a = saturate(blob + ring);
                float3 c = i.col.rgb * 3.0 * ring / max(a, 1e-4);
                UNITY_APPLY_FOG_COLOR(i.fogCoord, c, fixed4(0,0,0,0));
                return half4(c, a);
            }
            ENDCG
        }
    }
    Fallback Off
}
