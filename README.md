---
license: CC BY-NC-SA
technical_domain:
  - CV
  - 分割
---

<div><span style="font-size: 16px;">摘要</span></div>
<div>尽管 RGB-D 传感器在多项视觉任务（例如 3D 重建）方面取得了重大突破，但我们还没有在高级场景理解方面实现类似的性能提升。可能造成这种情况的主要原因之一是缺乏合理大小的基准，其中包含用于训练的 3D 注释和用于评估的 3D 指标。在本文中，我们提出了一个 RGB-D 基准套件，旨在推进所有主要场景理解任务的最新技术水平。我们的数据集由四个不同的传感器捕获，包含 10,000 个 RGB-D 图像，其规模与 PASCAL VOC 相似。整个数据集经过密集注释，包括 146,617 个 2D 多边形和 58,657 个具有准确对象方向的 3D 边界框，以及场景的 3D 房间布局和类别。</div>
<div>&nbsp;</div>
<div>
<div><span style="font-size: 16px;">数据和注释</span><br/>SUNRGBD V1：此文件包含SUNRGBD V1的 10335 个 RGBD 图像。<br/>该数据集包含来自纽约大学深度 v2 [1]、伯克利 B3DO [2] 和 SUN3D [3] 的 RGB-D 图像。除了这篇论文，如果你使用这个数据集，你还需要引用以下论文。[1] N. Silberman、D. Hoiem、P. Kohli、R. Fergus。室内分割和支持从 rgbd 图像推断。在 ECCV，2012 年。</div>
<div>[2] A. Janoch、S. Karayev、Y. Jia、JT Barron、M. Fritz、K. Saenko 和 T. Darrell。类别级 3-d 对象数据集：让 kinect 发挥作用。在 ICCV 计算机视觉消费者深度相机研讨会上，2011 年。</div>
<div>[3] J. Xiao、A. Owens 和 A. Torralba。SUN3D：使用 SfM 和对象标签重建的大空间数据库。在 ICCV, 2013</div>
<div>SUNRGBDtoolbox：此文件包含用于加载和可视化数据的注释和 Matlab 代码。</div>
</div>
<div>&nbsp;</div>
<div><span style="font-size: 16px;">致谢</span><br/>这项工作得到了英特尔公司捐赠基金的支持。我们感谢 Thomas Funkhouser、Jitendra Malik、Alexi A. Efros 和 Szymon Rusinkiewicz 的宝贵讨论。我们还要感谢 Linguang Zhang、Fisher Yu、Yinda Zhang、Luna Song、Pingmei Xu 和 Guoxuan Zhang 的捕获和标记。</div>
<div>&nbsp;</div>
<div><span style="font-size: 16px;">参考</span><br/>[4] B. Zhou, A. Lapedriza, J. Xiao, A. Torralba, and A. Oliva Learning Deep Features for Scene Recognition using Places Database Advances in Neural Information Processing Systems 27 (NIPS2014)</div>
<div>&nbsp;</div>
<div>来源：https://rgbd.cs.princeton.edu/</div>