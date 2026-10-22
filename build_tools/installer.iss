; ImDeploy 安装脚本（由 build_windows.py 生成）
#define MyAppName "ImDeploy 本地模型部署器"
#define MyAppVersion "0.1.0"
#define MyAppExeName "imdeploy.exe"

[Setup]
AppId={{8F2A6C31-4D7B-4E2A-9C51-3B7E5D9A0001}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
DefaultDirName={autopf}\ImDeploy
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=F:\vibe coding工程\imdeploy\dist\installer
OutputBaseFilename=imdeploy-setup-0.1.0
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加任务:"

[Files]
Source: "F:\vibe coding工程\imdeploy\dist\imdeploy\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
Name: "{app}\models"
Name: "{app}\data"
Name: "{app}\logs"

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "立即启动"; Flags: nowait postinstall skipifsilent
