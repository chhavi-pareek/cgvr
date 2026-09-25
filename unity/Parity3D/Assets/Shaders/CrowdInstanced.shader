// Built-In Render Pipeline, GPU-instanced crowd body parts. Lit by the scene's sun with its
// shadows, ambient from the sky, a rim term so figures read against the floor, and an
// emissive "heat" term that makes accumulated divergence glow. Colour and heat are per
// instance, so a whole LOD stream is one draw call per 511 parts.
Shader "Parity/CrowdInstanced"
{
    Properties
    {
        _BaseColor ("Base Colour", Color) = (1,1,1,1)
        _Heat ("Heat", Float) = 0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        Pass
        {
            Tags { "LightMode"="ForwardBase" }
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing
            #pragma multi_compile_fwdbase nolightmap nodirlightmap nodynlightmap novertexlight
            #pragma multi_compile_fog
            #pragma target 3.5
            #include "UnityCG.cginc"
            #include "Lighting.cginc"
            #include "AutoLight.cginc"

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
                float3 wpos : TEXCOORD1;
                half4 col : COLOR;
                half heat : TEXCOORD2;
                SHADOW_COORDS(3)
                UNITY_FOG_COORDS(4)
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
                o.nrm = UnityObjectToWorldNormal(v.normal);
                o.wpos = mul(unity_ObjectToWorld, v.vertex).xyz;
                o.col = UNITY_ACCESS_INSTANCED_PROP(Props, _BaseColor);
                o.heat = UNITY_ACCESS_INSTANCED_PROP(Props, _Heat);
                TRANSFER_SHADOW(o);
                UNITY_TRANSFER_FOG(o, o.pos);
                return o;
            }

            half4 frag (v2f i) : SV_Target
            {
                float3 n = normalize(i.nrm);
                float3 l = _WorldSpaceLightPos0.xyz;
                float3 v = normalize(_WorldSpaceCameraPos - i.wpos);
                UNITY_LIGHT_ATTENUATION(atten, i, i.wpos);
                float ndl = saturate(dot(n, l));
                float3 amb = ShadeSH9(float4(n, 1));
                float rim = pow(1.0 - saturate(dot(n, v)), 3.0);
                float3 hv = normalize(l + v);
                float3 c = i.col.rgb * (amb + _LightColor0.rgb * ndl * atten);
                c += _LightColor0.rgb * pow(saturate(dot(n, hv)), 24.0) * 0.06 * atten;
                c += amb * rim * 0.35;
                c += float3(1.0, 0.16, 0.05) * i.heat * (0.5 + 2.0 * rim);
                UNITY_APPLY_FOG(i.fogCoord, c);
                return half4(c, 1);
            }
            ENDCG
        }

        Pass
        {
            Tags { "LightMode"="ShadowCaster" }
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_shadowcaster
            #pragma multi_compile_instancing
            #include "UnityCG.cginc"

            struct v2f { V2F_SHADOW_CASTER; };

            v2f vert (appdata_base v)
            {
                v2f o;
                UNITY_SETUP_INSTANCE_ID(v);
                TRANSFER_SHADOW_CASTER_NORMALOFFSET(o)
                return o;
            }

            float4 frag (v2f i) : SV_Target { SHADOW_CASTER_FRAGMENT(i) }
            ENDCG
        }
    }
    Fallback Off
}
