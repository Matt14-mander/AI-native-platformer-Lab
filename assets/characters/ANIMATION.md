# 角色动画

游戏读取 `child-with-puppy.animation.json`，以脚底中心为锚点。动画只改变显示，不改变碰撞盒或操作响应。

- 待机：120 tick 呼吸周期，幅度 1px。
- 行走 / 奔跑：复用两张迈步与待机姿势组成四拍循环，每移动 22px 切换一拍。
- 起跳 / 顶点 / 下落：按垂直速度切换，顶点范围 ±1.5。
- 落地：新增屈膝图，8 tick 内逐渐回弹；再次起跳立即打断。
- 刹车 / 掉头：新增后仰图，按速度降低或朝向与速度相反触发。
- 受伤：复用下落图加倾斜反馈，保留无敌闪烁；死亡固定倾斜姿势。
- 暂停冻结动画；重开清空动画状态。左右方向镜像，变换后的帧缓存复用。

## 新素材生成记录

工具：内置 imagegen；参考图：`child-with-puppy-idle.png`。以下是最终提示词，输出保持透明背景，源图未覆盖。

### child-with-puppy-land.png

Use case: identity-preserve. Create ONE transparent game character sprite, landing crouch pose, using the supplied existing character as exact identity and painterly pastel style reference. Same boy hugging same cream puppy with dark floppy ears, charcoal hoodie, muted blue trousers, cream shoes, face looking right in same three-quarter side view. Full body visible. Change pose only: knees bent visibly to absorb landing, torso lowered, both feet planted, puppy held securely close. Keep head/body proportions, costume colors, face and rough painted edges consistent. No floor, shadow, scenery, lettering or extra objects. Actual transparent background, single centered character with margin. This is a new sibling game animation frame.

### child-with-puppy-brake.png

Use case: identity-preserve. ONE transparent platform-game sprite, braking/skidding pose. Input is exact character identity/style reference. Preserve boy hugging cream puppy with dark floppy ears, charcoal hoodie, muted blue trousers, cream shoes, big dark hair and vertical black eyes, rough soft painted pastel texture. Same right-facing three-quarter view, full body centered with margin. Pose only changes: boy leans torso backward to the left, extends right foot forward to the right with heel planted, bends rear knee, hugs puppy firmly. Clear readable silhouette of slowing down while moving right. Keep proportions and head size consistent. No motion lines, floor, dust, shadows, scenery, text or extra objects. Actual transparent background.
