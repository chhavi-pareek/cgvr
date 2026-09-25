// Built-In Render Pipeline, GPU-instanced, per-instance colour. Deliberately plain: the
// interesting GPU work in this project is the allocator, not the shading, and a shader with
// no pipeline-specific includes is one less thing to debug.
Shader "Parity/CrowdInstanced"
{
    Properties
    {
        _BaseColor ("Base Colour", Color) = (1,1,1,1)
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing
            #pragma target 3.0
            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float3 nrm : TEXCOORD0;
                fixed4 col : COLOR;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            UNITY_INSTANCING_BUFFER_START(Props)
                UNITY_DEFINE_INSTANCED_PROP(float4, _BaseColor)
            UNITY_INSTANCING_BUFFER_END(Props)

            v2f vert (appdata v)
            {
                v2f o;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_TRANSFER_INSTANCE_ID(v, o);
                o.pos = UnityObjectToClipPos(v.vertex);
                o.nrm = UnityObjectToWorldNormal(v.normal);
                o.col = UNITY_ACCESS_INSTANCED_PROP(Props, _BaseColor);
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                UNITY_SETUP_INSTANCE_ID(i);
                float3 l = normalize(float3(0.35, 0.9, 0.25));
                float ndl = saturate(dot(normalize(i.nrm), l)) * 0.75 + 0.25;
                return fixed4(i.col.rgb * ndl, 1);
            }
            ENDCG
        }
    }
    Fallback Off
}
