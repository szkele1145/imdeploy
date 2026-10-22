; ImDeploy 安装脚本（由 build_windows.py 生成，全中文）
; 编码必须为 UTF-8 BOM，否则 Inno Setup 读中文会乱码
#define MyAppName "ImDeploy 本地模型部署器"
#define MyAppVersion "0.1.0"
#define MyAppExeName "imdeploy.exe"
#define MyAppPublisher "ImDeploy"

[Setup]
AppId={{8F2A6C31-4D7B-4E2A-9C51-3B7E5D9A0001}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
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
; 安装界面语言：简体中文
SetupIconFile=F:\vibe coding工程\imdeploy\build_tools\app.ico
UninstallDisplayName={#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}
VersionInfoDescription=ImDeploy 本地模型部署器 安装程序
VersionInfoProductName={#MyAppName}
VersionInfoCompany={#MyAppPublisher}
VersionInfoTextVersion={#MyAppVersion}

[Languages]
Name: "chs"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加任务："; Flags: checkedonce

[Files]
Source: "F:\vibe coding工程\imdeploy\dist\imdeploy\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
Name: "{app}\models"; Permissions: users-modify
Name: "{app}\data"; Permissions: users-modify
Name: "{app}\logs"; Permissions: users-modify

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\app.ico"
Name: "{group}\卸载 {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\app.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "立即启动 {#MyAppName}"; Flags: nowait postinstall skipifsilent

[Messages]
; 覆盖几处关键提示，确保全中文
WelcomeLabel1=欢迎使用 [name] 安装向导
WelcomeLabel2=即将在你的电脑上安装 [name/ver]。%n%n这个程序会按你的硬件推荐并下载本地大模型，提供 OpenAI 兼容接口。%n%n点“下一步”继续。
FinishedHeadingLabel=安装完成
FinishedLabelNoIcons=[name] 已安装完成。
FinishedLabel=[name] 已安装完成。可以选择下方的勾选框直接启动。
