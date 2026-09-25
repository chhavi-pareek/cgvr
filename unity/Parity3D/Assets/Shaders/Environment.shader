// Every surface of the three sets: one surface shader whose patterns are computed from world
// position, so there are no textures to ship and detail holds at any camera distance. Lit by
// the Standard model, so it takes the sun, its shadows, sky ambient and sky reflections.
//
//   0 plain      1 floor tiles    2 plaza paving   3 building facade   4 hazard stripes
//   5 planting   6 asphalt        7 wall tiles     8 timber            9 glazing
Shader "Parity/Environment"
{
    Properties
    {
        _ColorA ("A", Color) = (0.6,0.6,0.6,1)
        _ColorB ("B", Color) = (0.5,0.5,0.5,1)
        _ColorC ("C", Color) = (0.1,0.1,0.1,1)
        _Pattern ("Pattern", Float) = 0
        _Scale ("Scale", Vector) = (1,1,0.03,0)
        _Gloss ("Smoothness", Range(0,1)) = 0.2
        _Metal ("Metallic", Range(0,1)) = 0
        _Emission ("Emission", Color) = (0,0,0,1)
        _Glow ("Window Glow", Color) = (0,0,0,1)
        _Center ("Centre", Vector) = (0,0,0,0)
        _Seed ("Seed", Float) = 0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        CGPROGRAM
        #pragma surface surf Standard fullforwardshadows addshadow
        #pragma target 3.5

        half4 _ColorA, _ColorB, _ColorC, _Emission, _Glow;
        float4 _Scale, _Center;
        half _Gloss, _Metal;
        float _Pattern, _Seed;

        struct Input { float3 worldPos; float3 worldNormal; };

        float hash21(float2 p)
        {
            p = frac(p * float2(123.34, 456.21) + _Seed * 0.618);
            p += dot(p, p + 45.32);
            return frac(p.x * p.y);
        }
        float vnoise(float2 p)
        {
            float2 i = floor(p), f = frac(p);
            float2 u = f * f * (3 - 2 * f);
            return lerp(lerp(hash21(i), hash21(i + float2(1, 0)), u.x),
                        lerp(hash21(i + float2(0, 1)), hash21(i + float2(1, 1)), u.x), u.y);
        }
        float fbm(float2 p) { return vnoise(p) * 0.55 + vnoise(p * 2.1) * 0.3 + vnoise(p * 4.3) * 0.15; }

        // grout mask for a cell grid: 1 inside the tile, 0 on the joint
        float joint(float2 f, float w)
        {
            float2 e = min(f, 1 - f);
            return smoothstep(0, w, min(e.x, e.y));
        }

        void surf (Input IN, inout SurfaceOutputStandard o)
        {
            float3 wp = IN.worldPos;
            float3 an = abs(IN.worldNormal);
            bool up = an.y > 0.5;
            // on a wall, u runs along the face and v is height
            float2 wall = float2(an.x > an.z ? wp.z : wp.x, wp.y);
            float2 flat = wp.xz;
            float2 st = up ? flat : wall;
            int pat = (int)round(_Pattern);

            half3 alb = _ColorA.rgb;
            half gloss = _Gloss, metal = _Metal;
            half3 emit = _Emission.rgb;

            if (pat == 0)
            {
                alb *= 0.9 + 0.2 * fbm(st * 1.7);
            }
            else if (pat == 1)
            {
                float2 uv = flat / _Scale.xy;
                float2 id = floor(uv);
                float g = joint(frac(uv), _Scale.z);
                float h = hash21(id);
                float vein = smoothstep(0.55, 0.9, fbm(flat * 0.9 + h * 7) * (0.8 + 0.4 * sin(flat.x * 2.3 + fbm(flat * 0.4) * 6)));
                half3 tile = lerp(_ColorA.rgb, _ColorB.rgb, h) * (0.94 + 0.12 * vnoise(flat * 3));
                tile = lerp(tile, tile * 0.82, vein * 0.5);
                alb = lerp(_ColorC.rgb, tile, g);
                gloss = lerp(0.08, _Gloss * (0.9 + 0.1 * h), g);
            }
            else if (pat == 2)
            {
                float2 uv = flat / _Scale.xy;
                float row = floor(uv.y);
                uv.x += 0.5 * fmod(abs(row), 2);
                float2 id = floor(uv);
                float g = joint(frac(uv), _Scale.z);
                float h = hash21(id);
                half3 paver = lerp(_ColorA.rgb, _ColorB.rgb, h) * (0.9 + 0.2 * fbm(flat * 2.5));
                float2 d = flat - _Center.xy;
                float r = length(d);
                float a = atan2(d.y, d.x);
                // granite bands every 12 m and a compass inlay at the centre
                float band = step(frac(r / 12.0), 0.035) * step(14.0, r);
                float star = 9.5 * (0.28 + 0.72 * pow(abs(cos(a * 4.0)), 8.0));
                float inStar = step(r, star);
                float half_ = step(0.5, frac(a / 0.7853982 + 0.5 * step(0, sin(a * 4.0 + 1.5707963))));
                float ringLine = step(abs(r - 10.2), 0.18) + step(abs(r - 11.0), 0.08);
                half3 dark = _ColorC.rgb * (0.9 + 0.2 * fbm(flat * 3));
                paver = lerp(paver, lerp(paver, dark, 0.45), band);
                paver = lerp(paver, lerp(_ColorA.rgb * 1.08, dark * 1.3, half_), inStar);
                paver = lerp(paver, dark, saturate(ringLine));
                alb = lerp(_ColorC.rgb * 0.8, paver, lerp(g, 1, saturate(inStar + ringLine)));
                gloss = lerp(0.05, _Gloss, g) + 0.25 * inStar;
            }
            else if (pat == 3)
            {
                if (up)
                {
                    alb = _ColorC.rgb * (0.8 + 0.4 * fbm(flat * 2));
                    gloss = 0.1;
                }
                else
                {
                    float bay = _Scale.x, storey = _Scale.y, ground = _Scale.z;
                    float2 f;
                    float2 id;
                    bool win;
                    if (wall.y < ground)
                    {
                        // shopfronts: wide lit glazing over a stone plinth
                        float u = wall.x / (bay * 2.0);
                        id = float2(floor(u), -1);
                        f = float2(frac(u), wall.y / ground);
                        win = f.x > 0.08 && f.x < 0.92 && f.y > 0.12 && f.y < 0.82;
                    }
                    else
                    {
                        float2 uv = float2(wall.x / bay, (wall.y - ground) / storey);
                        id = floor(uv);
                        f = frac(uv);
                        win = f.x > 0.2 && f.x < 0.8 && f.y > 0.22 && f.y < 0.86;
                    }
                    float h = hash21(id + _Seed);
                    half3 wallc = _ColorA.rgb * (0.88 + 0.24 * fbm(wall * 0.8));
                    // a cornice line at every storey
                    float storeyLine = wall.y > ground ? step(frac((wall.y - ground) / storey), 0.05) : 0;
                    wallc = lerp(wallc, _ColorB.rgb, storeyLine);
                    if (win)
                    {
                        alb = _ColorC.rgb;
                        gloss = 0.92;
                        metal = 0.2;
                        float lit = step(0.62, h) + (wall.y < ground ? 0.8 : 0);
                        float sill = smoothstep(0.2, 0.9, f.y);
                        emit += _Glow.rgb * saturate(lit) * (0.55 + 0.45 * hash21(id * 1.7)) * (0.7 + 0.3 * sill);
                    }
                    else alb = wallc;
                }
            }
            else if (pat == 4)
            {
                float t = frac((st.x + st.y) / _Scale.x);
                alb = t < 0.5 ? _ColorA.rgb : _ColorC.rgb;
                alb *= 0.9 + 0.15 * fbm(st * 4);
            }
            else if (pat == 5)
            {
                float n = fbm(st * 3.5);
                alb = lerp(_ColorA.rgb, _ColorB.rgb, n) * (0.8 + 0.4 * vnoise(st * 17));
                gloss = 0.05;
            }
            else if (pat == 6)
            {
                float n = fbm(flat * 1.2);
                alb = _ColorA.rgb * (0.8 + 0.35 * n) + step(0.985, hash21(floor(flat * 40))) * 0.08;
                gloss = 0.12 + 0.2 * n;
            }
            else if (pat == 7)
            {
                float dado = _Center.w;
                if (up) alb = _ColorC.rgb * 1.25 * (0.9 + 0.2 * fbm(flat * 2));
                else if (wall.y < dado)
                {
                    alb = _ColorB.rgb * (0.9 + 0.15 * fbm(wall * 2));
                    gloss = 0.35;
                }
                else
                {
                    float2 uv = wall / _Scale.xy;
                    uv.x += 0.5 * fmod(abs(floor(uv.y)), 2);
                    float g = joint(frac(uv), _Scale.z);
                    alb = lerp(_ColorC.rgb, _ColorA.rgb * (0.95 + 0.08 * hash21(floor(uv))), g);
                    gloss = lerp(0.1, _Gloss, g);
                    alb = lerp(alb, _ColorC.rgb * 1.4, step(abs(wall.y - dado - 0.06), 0.05));
                }
            }
            else if (pat == 8)
            {
                float grain = sin((wp.x + wp.z) * 38 + fbm(st * 3) * 9);
                alb = _ColorA.rgb * (0.82 + 0.12 * grain + 0.1 * vnoise(st * 9));
                gloss = 0.3;
            }
            else if (pat == 9)
            {
                // curtain wall: mullion grid over glazing that glows with the daylight beyond
                float2 uv = wall / _Scale.xy;
                float2 f = frac(uv);
                float m = 1 - joint(f, _Scale.z);
                alb = lerp(_ColorA.rgb, _ColorC.rgb, m);
                gloss = lerp(0.95, 0.5, m);
                metal = lerp(0.1, 0.8, m);
                emit += _Glow.rgb * (1 - m) * (0.85 + 0.15 * f.y);
            }

            o.Albedo = alb;
            o.Smoothness = gloss;
            o.Metallic = metal;
            o.Emission = emit;
        }
        ENDCG
    }
    Fallback "Diffuse"
}
