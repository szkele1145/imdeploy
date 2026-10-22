package com.imdeploy.app;

import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.KeyEvent;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.FrameLayout;

/**
 * ImDeploy 安卓壳。
 *
 * 首次启动引导安装 Termux 并执行随包脚本；
 * 之后 WebView 指向本机 API 面板：http://127.0.0.1:8080
 */
public class MainActivity extends Activity {

    private WebView web;
    private static final String PANEL_URL = "http://127.0.0.1:8080";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        FrameLayout root = new FrameLayout(this);
        web = new WebView(this);
        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        s.setAllowFileAccess(false);
        s.setMixedContentMode(WebSettings.MIXED_CONTENT_ALWAYS_ALLOW);
        web.setWebViewClient(new WebViewClient());

        root.addView(web);
        setContentView(root);

        if (!isTermuxInstalled()) {
            promptInstallTermux();
        } else {
            launchTermuxScript("install-termux.sh");
        }
        web.loadUrl(PANEL_URL);
    }

    private boolean isTermuxInstalled() {
        try {
            getPackageManager().getPackageInfo("com.termux", 0);
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    private void promptInstallTermux() {
        try {
            startActivity(new Intent(Intent.ACTION_VIEW,
                    Uri.parse("https://f-droid.org/packages/com.termux/")));
        } catch (ActivityNotFoundException ignored) {
        }
    }

    /** 通过 RUN_COMMAND intent 让 Termux 执行随包脚本。 */
    private void launchTermuxScript(String script) {
        try {
            Intent i = new Intent();
            i.setClassName("com.termux", "com.termux.app.RunCommandService");
            i.setAction("com.termux.RUN_COMMAND");
            i.putExtra("com.termux.RUN_COMMAND_PATH",
                    "/data/data/com.termux/files/usr/bin/bash");
            i.putExtra("com.termux.RUN_COMMAND_ARGUMENTS",
                    new String[]{"-c",
                        "mkdir -p ~/imdeploy && cd ~/imdeploy && "
                        + "cp -r /data/data/com.imdeploy.app/files/termux/* ~/imdeploy/ 2>/dev/null || true; "
                        + "bash ~/imdeploy/" + script});
            i.putExtra("com.termux.RUN_COMMAND_WORKDIR",
                    "/data/data/com.termux/files/home");
            i.putExtra("com.termux.RUN_COMMAND_BACKGROUND", false);
            startService(i);
        } catch (Exception ignored) {
        }
    }

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        if (keyCode == KeyEvent.KEYCODE_BACK && web != null && web.canGoBack()) {
            web.goBack();
            return true;
        }
        return super.onKeyDown(keyCode, event);
    }

    @Override
    protected void onDestroy() {
        if (web != null) {
            web.destroy();
            web = null;
        }
        super.onDestroy();
    }
}
