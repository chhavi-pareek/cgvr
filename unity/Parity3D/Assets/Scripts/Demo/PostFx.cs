// Bloom, ACES tone mapping, grade and vignette for one camera, without the post-processing
// package. Both panels get the same settings, so the finish is never a difference between them.
using UnityEngine;

namespace Parity
{
    [RequireComponent(typeof(Camera))]
    public sealed class PostFx : MonoBehaviour
    {
        public static bool On = true;
        public float Exposure = 1.0f, Bloom = 0.5f, Threshold = 1.0f, Vignette = 0.3f;
        public float Saturation = 1.06f, Contrast = 1.05f;
        public Color Tint = Color.white;

        static Material mat;
        readonly RenderTexture[] mips = new RenderTexture[6];

        void OnRenderImage(RenderTexture src, RenderTexture dst)
        {
            if (!On || !Ready()) { Graphics.Blit(src, dst); return; }
            Process(src, dst);
        }

        static bool Ready()
        {
            if (mat != null) return true;
            var sh = Shader.Find("Hidden/Parity/PostFx");
            if (sh == null) return false;
            mat = new Material(sh) { hideFlags = HideFlags.HideAndDontSave };
            return true;
        }

        public void Process(RenderTexture src, RenderTexture dst)
        {
            if (!Ready()) { Graphics.Blit(src, dst); return; }
            mat.SetFloat("_Threshold", Threshold);
            mat.SetFloat("_BloomStrength", Bloom);
            mat.SetFloat("_Exposure", Exposure);
            mat.SetFloat("_Vignette", Vignette);
            mat.SetFloat("_Saturation", Saturation);
            mat.SetFloat("_Contrast", Contrast);
            mat.SetColor("_Tint", Tint);

            int w = Mathf.Max(src.width / 2, 1), h = Mathf.Max(src.height / 2, 1), n = 0;
            mips[n] = RenderTexture.GetTemporary(w, h, 0, RenderTextureFormat.DefaultHDR);
            Graphics.Blit(src, mips[n++], mat, 0);
            while (n < mips.Length && w > 8 && h > 8)
            {
                w /= 2; h /= 2;
                mips[n] = RenderTexture.GetTemporary(w, h, 0, RenderTextureFormat.DefaultHDR);
                Graphics.Blit(mips[n - 1], mips[n], mat, 1);
                n++;
            }
            for (int i = n - 1; i > 0; i--) Graphics.Blit(mips[i], mips[i - 1], mat, 2);
            mat.SetTexture("_Bloom", mips[0]);
            Graphics.Blit(src, dst, mat, 3);
            for (int i = 0; i < n; i++) { RenderTexture.ReleaseTemporary(mips[i]); mips[i] = null; }
        }
    }
}
