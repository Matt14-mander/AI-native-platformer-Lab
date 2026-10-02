# Original Asset Workspace

本目录用于未来可发布的原创或有明确授权的资产。

- `characters/`：角色 sprite、动画定义；
- `tilesets/`：地形、机关、背景；
- `audio/`：音乐和音效；
- `themes/`：主题 manifest 与资源映射。

原项目的马里奥相关素材继续保留在 `resources/`，不得复制到发布资产包。

## 湖畔粉彩主题（第一批原创素材）

参考用户提供的头像，统一采用雾蓝天空、奶油色高光、深棕轮廓与手绘粉彩笔触。每张素材均由内置 imagegen 单独生成，PNG 原图保留在本目录中。

| 用途 | 文件 | 建议游戏显示尺寸 |
| --- | --- | --- |
| 主角站姿（朝右，抱小狗） | characters/child-with-puppy-idle.png | 高 48–56 px；按透明内容边界缩放 |
| 湖畔远景 | backgrounds/lakeside-day.png | 背景层，按屏幕高缩放并水平裁切/重复 |
| 苔土地面 | tilesets/moss-earth-ground.png | 地面水平段，高约 60 px |
| 树桩障碍 | tilesets/mossy-stump-obstacle.png | 对应现有关卡的竖向障碍矩形 |
| 苔土台阶 | tilesets/mossy-step-block.png | 对应 40×44 px 台阶 |
| 琥珀橡果 | tilesets/amber-acorn-collectible.png | 对应 16×24 px 收集物 |

这些独立原始图已接入当前可玩关卡。运行时按透明像素边界裁切，并以关卡碰撞矩形为准绘制地面、树桩和台阶；角色显示高 52 px，底部中心与 24×32 px 碰撞体对齐。

生成提示词要点：以头像为风格和色板参考；独立生成侧面抱小狗的主角、无前景的湖畔远景、苔土横向地面、可站立树桩、方形台阶与发光橡果；使用 gouache / oil-pastel 手绘质感；前景素材要求透明背景、无文字和多余角色。

## 主角基础动作

右向角色动作保存在 characters/：idle、run-1、run-2、jump、fall。状态与播放建议见 characters/child-with-puppy.animation.json。向左可水平翻转；渲染时裁切透明边并以底部中心对齐，避免帧切换时位置跳动。动作 PNG 已接入运行时渲染器：根据水平速度和垂直速度切换姿态。

## 新增互动素材

- enemies/shadow-hedgehog.png：地面巡逻刺猬，左右方向运行时翻转；
- tilesets/cracked-clay-brick.png：可从下方顶破的苔土砖；
- tilesets/acorn-reward-box.png：一次性奖励箱，使用后在画面中变暗；
- items/blue-ward-berry.png：可拾取的护盾浆果，抵挡一次敌人碰撞。

以上图片由内置 imagegen 分别生成，以原头像的深棕轮廓、雾蓝与奶油色手绘粉彩为风格参考；均为透明背景单体 PNG。第一关前段的实体与奖励由 game_content/levels/level_1.json 配置，碰撞和状态由无画面核心处理。
