# 湖岸场景图层

当前渲染器已接入独立图层。原始 lakeside-day.png 保留，天空从原图上部取样。

| 层 | 文件 | 镜头速度比例 |
| --- | --- | --- |
| 天空 | lakeside-day.png 上部 | 0.04 |
| 远山 | lakeside-mountains.png | 0.12 |
| 湖面 | lakeside-water.png | 0.32 |
| 近岸 | lakeside-nearshore.png | 0.62 |
| 可游玩平台 | 原 tilesets 素材 | 1.00 |

图层以正反交替平铺使接缝两侧像素一致。环境效果依赖模拟 tick：至多 18 处低亮度波光、5 只微光萤火，草叶只出现在未被平台覆盖的地面上。暂停时画面冻结；镜头移动不改变碰撞规则。可行走顶边的 1px 浅色高光与实际碰撞顶面同高，下面是 4px 深色苔边；堆叠平台内部不重复绘制顶边，悬崖两端截断草叶，树桩保留木质侧面、台阶保留岩土侧面。

三个新增 PNG 使用内置 imagegen 生成；输入参考均为 lakeside-day.png，远山与近岸保留 alpha，湖面不透明。最终提示词如下。

## 远山

Create a wide 3:1 horizontal game parallax layer using the reference's exact rough gouache / oil-pastel hand painted lakeside style, desaturated sage and blue-green palette. ONLY distant rolling mountains and very distant forest silhouettes. Entire top 35 percent transparent sky, irregular soft mountain skyline around middle, opaque wooded mountain bases filling bottom edge. No sky paint, clouds, water, buildings, platforms, text or characters. Low contrast hazy hills, quiet scenery. Hills extend continuously to both horizontal edges with matching height and color so layer tiles horizontally. Genuine transparent background above mountains. Reference is style inspiration, create a separate production layer.

## 湖面

Create a wide 3:1 horizontal tileable game background layer of ONLY calm lake water, filling the whole rectangle edge to edge, opaque. Match reference rough gouache / oil-pastel texture, muted blue teal lake with soft horizontal sage blue strokes and a few understated cream reflected cloud patches. View almost level across lake, flat horizontal ripples, distant water slightly lighter top, near water darker bottom. NO horizon, shoreline, mountains, sky, plants, boats, text or characters. Low contrast, no large focal detail. Left and right edges have matching colors for horizontal tiling. Production parallax water layer.

## 近岸

Create a wide 3:1 transparent game parallax near-shore decoration strip matching reference rough gouache oil-pastel lakeside painting. A very low uneven ribbon of muted sage reeds, tufts of bank grass and a few soft gray-green pebbles, all along bottom quarter of canvas. Upper 70 percent is actual empty transparent space. Dark desaturated green silhouette, restrained texture and soft edges. NO continuous horizontal platform or walkable ledge, no dirt cross section, NO trees, tall plants, sky, water, hills, characters or text. Tufts stay short and irregular, extend to left/right borders with similar heights for repeating panorama. This is distant near-shore scenery BEHIND playable ground, not collision geometry. Genuine alpha transparency above and between vegetation.
