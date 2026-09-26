// Standalone build of the frame bench, for a machine without Unity.
//
//   Unity -batchmode -quit -projectPath <p> -executeMethod ParityBuild.Bench [-target win64|mac] [-out <dir>]
//
// The crowd, set and post shaders are only ever found by name (Shader.Find), so nothing in a
// scene references them and a player build would strip them: they go into Always Included. The
// player itself is an empty scene; ParityFrameBench starts on -parityBench.
using System;
using System.IO;
using System.Reflection;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;

public static class ParityBuild
{
    static readonly string[] Shaders =
    {
        "Parity/CrowdInstanced", "Parity/CrowdImpostor", "Parity/CrowdDecal", "Parity/Environment", "Parity/Sign",
        "Hidden/Parity/PostFx", "Skybox/Procedural",
    };
    const string Scene = "Assets/Scenes/ParityBench.unity";

    public static void Bench()
    {
        bool mac = Arg("-target") == "mac";
        var target = mac ? BuildTarget.StandaloneOSX : BuildTarget.StandaloneWindows64;
        string outDir = Path.GetFullPath(Arg("-out") ?? Path.Combine("Builds", mac ? "mac" : "win64"));
        int failures = IncludeShaders();

        if (!File.Exists(Scene))
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Scene));
            var s = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            EditorSceneManager.SaveScene(s, Scene);
        }
        PlayerSettings.productName = "ParityBench";
        PlayerSettings.runInBackground = true;
        PlayerSettings.fullScreenMode = FullScreenMode.Windowed;
        PlayerSettings.defaultScreenWidth = 960;
        PlayerSettings.defaultScreenHeight = 540;
        PlayerSettings.SetScriptingBackend(NamedBuildTarget.Standalone, ScriptingImplementation.Mono2x);
        if (!mac && !SetWindowsArchitecture("x64")) failures++;

        if (Directory.Exists(outDir)) Directory.Delete(outDir, true);
        var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
        {
            scenes = new[] { Scene },
            locationPathName = Path.Combine(outDir, mac ? "ParityBench.app" : "ParityBench.exe"),
            target = target,
            options = BuildOptions.None,
        });
        var sum = report.summary;
        Debug.Log($"[ParityBuild] {sum.result}: {sum.outputPath}, {sum.totalSize / (1024 * 1024)} MB, {sum.totalErrors} errors");
        if (sum.result != BuildResult.Succeeded) failures++;
        EditorApplication.Exit(failures == 0 ? 0 : 1);
    }

    // On an Apple Silicon editor the Windows player defaults to ARM64, which an ordinary x64 PC
    // refuses to start ("not compatible with the version of Windows"). The setting lives in the
    // Windows build module, so it is set by reflection: the editor scripts still compile where
    // that module is not installed.
    static bool SetWindowsArchitecture(string arch)
    {
        var settings = Type.GetType("UnityEditor.WindowsStandalone.UserBuildSettings, UnityEditor.WindowsStandalone.Extensions");
        var prop = settings?.GetProperty("architecture", BindingFlags.Public | BindingFlags.Static);
        if (prop == null) { Debug.LogError("[ParityBuild] Windows build module missing: cannot set the player architecture"); return false; }
        prop.SetValue(null, Enum.Parse(prop.PropertyType, arch));
        Debug.Log($"[ParityBuild] Windows player architecture {prop.GetValue(null)}");
        return true;
    }

    static int IncludeShaders()
    {
        var so = new SerializedObject(GraphicsSettings.GetGraphicsSettings());
        var list = so.FindProperty("m_AlwaysIncludedShaders");
        int missing = 0;
        foreach (var name in Shaders)
        {
            var sh = Shader.Find(name);
            if (sh == null) { Debug.LogError($"[ParityBuild] shader {name} not found"); missing++; continue; }
            bool has = false;
            for (int i = 0; i < list.arraySize && !has; i++) has = list.GetArrayElementAtIndex(i).objectReferenceValue == sh;
            if (has) continue;
            list.InsertArrayElementAtIndex(list.arraySize);
            list.GetArrayElementAtIndex(list.arraySize - 1).objectReferenceValue = sh;
        }
        // the crowd's materials are made at run time, so no build-time material asks for the
        // instancing variants and "strip unused" drops them: every crowd draw then silently vanishes
        so.FindProperty("m_InstancingStripping").intValue = 2;   // keep all
        so.ApplyModifiedProperties();
        AssetDatabase.SaveAssets();
        return missing;
    }

    static string Arg(string key)
    {
        var a = Environment.GetCommandLineArgs();
        int i = Array.IndexOf(a, key);
        return i >= 0 && i + 1 < a.Length ? a[i + 1] : null;
    }
}
