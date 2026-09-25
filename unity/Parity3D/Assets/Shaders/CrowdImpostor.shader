// Geometry tier 3: one camera-facing card per agent. The figure is an alpha-tested signed
// distance silhouette drawn in the card's own metres (uv = object xy), coloured shirt /
// trousers / skin, and shaded with a cylindrical fake normal so it does not read as flat.
Shader "Parity/CrowdImpostor"
{
    Properties
    {
        _BaseColor ("Shirt", Color) = (1,1,1,1)
        _Color2 ("Trousers", Color) = (0.2,0.2,0.3,1)
        _Color3 ("Skin", Color) = (0.7,0.5,0.4,1)
        _Heat ("Heat", Float) = 0
    }
    CGINCLUDE
    #include "UnityCG.cginc"
    float sdBox(float2 p, float2 c, float2 h)
    {
        float2 d = abs(p - c) - h;
        return length(max(d, 0)) + min(max(d.x, d.y), 0);
    }
    // returns distance; part: 0 shirt, 1 trousers, 2 skin, 3 hair
    float Figure(float2 p, out int part)
    {
        float head = length(p - float2(0, 1.635)) - 0.118;
        float torso = sdBox(p, float2(0, 1.19), float2(0.135, 0.27)) - 0.045;
        float arms = min(sdBox(p, float2(-0.215, 1.13), float2(0.03, 0.27)) - 0.025,
                         sdBox(p, float2( 0.215, 1.13), float2(0.03, 0.27)) - 0.025);
        float legs = min(sdBox(p, float2(-0.085, 0.46), float2(0.05, 0.42)) - 0.022,
                         sdBox(p, float2( 0.085, 0.46), float2(0.05, 0.42)) - 0.022);
        float body = min(torso, arms);
        float d = min(min(head, body), legs);
        part = legs <= d + 1e-4 ? 1 : (head <= d + 1e-4 ? (p.y > 1.67 ? 3 : 2) : 0);
        return d;
    }
    ENDCG
    SubShader
    {
        Tags { "RenderType"="TransparentCutout" "Queue"="AlphaTest" }
        Cull Off
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
            #include "Lighting.cginc"
            #include "AutoLight.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float2 uv : TEXCOORD0;
                float3 wpos : TEXCOORD1;
                float3 right : TEXCOORD2;
                float3 fwd : TEXCOORD3;
                half4 c1 : TEXCOORD4;
                half4 c2 : TEXCOORD5;
                half4 c3 : TEXCOORD6;
                SHADOW_COORDS(7)
                UNITY_FOG_COORDS(8)
            };

            UNITY_INSTANCING_BUFFER_START(Props)
                UNITY_DEFINE_INSTANCED_PROP(float4, _BaseColor)
                UNITY_DEFINE_INSTANCED_PROP(float4, _Color2)
                UNITY_DEFINE_INSTANCED_PROP(float4, _Color3)
                UNITY_DEFINE_INSTANCED_PROP(float, _Heat)
            UNITY_INSTANCING_BUFFER_END(Props)

            v2f vert (appdata v)
            {
                v2f o;
                UNITY_SETUP_INSTANCE_ID(v);
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = v.vertex.xy;
                o.wpos = mul(unity_ObjectToWorld, v.vertex).xyz;
                o.right = normalize(unity_ObjectToWorld._m00_m10_m20);
                o.fwd = normalize(unity_ObjectToWorld._m02_m12_m22);
                o.c1 = UNITY_ACCESS_INSTANCED_PROP(Props, _BaseColor);
                o.c2 = UNITY_ACCESS_INSTANCED_PROP(Props, _Color2);
                o.c3 = UNITY_ACCESS_INSTANCED_PROP(Props, _Color3);
                o.c1.a = UNITY_ACCESS_INSTANCED_PROP(Props, _Heat);
                TRANSFER_SHADOW(o);
                UNITY_TRANSFER_FOG(o, o.pos);
                return o;
            }

            half4 frag (v2f i) : SV_Target
            {
                int part;
                float d = Figure(i.uv, part);
                clip(-d);
                float3 alb = part == 0 ? i.c1.rgb : (part == 1 ? i.c2.rgb : i.c3.rgb);
                if (part == 3) alb = i.c3.rgb * 0.18;
                float nx = clamp(i.uv.x / 0.3, -1, 1);
                float3 n = normalize(i.right * nx + i.fwd * sqrt(saturate(1 - nx * nx)));
                float3 l = _WorldSpaceLightPos0.xyz;
                UNITY_LIGHT_ATTENUATION(atten, i, i.wpos);
                float3 amb = ShadeSH9(float4(n, 1));
                float3 c = alb * (amb + _LightColor0.rgb * saturate(dot(n, l)) * atten);
                c += float3(1.0, 0.16, 0.05) * i.c1.a * 1.2;
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

            struct appdata
            {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };
            struct v2f { V2F_SHADOW_CASTER; float2 uv : TEXCOORD1; };

            v2f vert (appdata v)
            {
                v2f o;
                UNITY_SETUP_INSTANCE_ID(v);
                o.uv = v.vertex.xy;
                TRANSFER_SHADOW_CASTER_NORMALOFFSET(o)
                return o;
            }

            float4 frag (v2f i) : SV_Target
            {
                int part;
                clip(-Figure(i.uv, part));
                SHADOW_CASTER_FRAGMENT(i)
            }
            ENDCG
        }
    }
    Fallback Off
}
