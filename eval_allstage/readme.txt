stage1: 输入：从表格中读取图片名称及对应的预测caption、参考caption。
        处理：分别对预测caption和参考caption进行拆分、匹配和验证。
        输出：把拆分、匹配和验证的结果保存到新的csv中。
stage2: 输入：图片及和stage1的输出结果。
        处理：计算caption的p、r和f1，计算风险等级的p、r和混淆矩阵。
        输出：caption的p、r和f1，计算风险等级的p、r和混淆矩阵。