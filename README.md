# math_ocr_agent
agentic method for math ocr project

## agent.py
对images文件夹中的单题图片和小图进行文本融合输出，图是预先处理好的，如果集成到线上，加载图片的位置做改动即可

单题图片是以`block`开头的文件，手写小图是以`crop`开头的文件，某些规则是基于这个写的，线上的大小图都是切题给的，可以忽略这里的规则

## vision_agent.py
对images文件夹中的单题图片进行vlm融合输出，逻辑是把我们自己模型的结果告诉其它vlm，然后让其最检查修改，这里没有小图的逻辑

## step1-parallel.py
这个是跑测试集的脚本，思路和vision_anget.py相同
