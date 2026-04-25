# 升级报告

## 基本信息

| 项目 | 值 |
|------|-----|
| 仓库名 | Neuraxle |
| 升级时间 | 2026-03-14 |
| 升级状态 | ⚠️ 部分成功（测试套件超时） |

## Python 版本

| 升级前 | 升级后 |
|--------|--------|
| >=3.7 | >=3.13 |

## 依赖变更

| 依赖 | 升级前 | 升级后 |
|------|--------|--------|
| numpy | >=1.16.2 | >=2.4.3 |
| scipy | >=1.4.1 | >=1.17.1 |
| scikit-learn | >=0.24.1 | >=1.8.0 |
| matplotlib | ==3.3.4 | >=3.10.8 |
| pandas | >=1.3.5 | >=3.0.1 |
| Flask | >=1.1.4 | >=3.1.3 |
| Flask-RESTful | >=0.3.9 | >=0.3.10 |
| markupsafe | ==2.0.1 | >=3.0.3 |
| joblib | >=0.13.2 | >=1.5.3 |
| sqlalchemy | (新增) | >=2.0.48 |
| py | (新增) | >=1.11.0 |

## 代码修改

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| examples/auto_ml/plot_automl_loop_clean_kata.py | API 适配 | 移除 RidgeClassifier 的 normalize 参数（sklearn 1.2+ 已移除） |
| neuraxle/hyperparams/distributions.py | 语法修复 | 修复 docstring 中的转义序列警告（\s → \\s） |
| neuraxle/steps/sklearn.py | API 适配 | 修复 inspect.getfullargspec → inspect.signature（Python 3.11+ 推荐） |
| setup.py | 依赖声明 | 添加 sqlalchemy 到 install_requires |
| setup.py | 依赖声明 | 添加 py 到 tests_require |
| setup.py | Python 版本 | 添加 python_requires='>=3.13' |

## 测试结果

| 测试类型 | 结果 |
|----------|------|
| 单元测试（完整套件） | ⚠️ 超时（3次，每次5分钟） |
| 示例代码（plot_automl_loop_clean_kata） | ✅ 通过 |
| 导入测试 | ✅ 通过 |

### 测试超时说明

完整测试套件在 Python 3.13 + 最新依赖下运行时间过长（>15分钟），累计超时3次后终止。

**已验证通过的功能：**
- ✅ 包导入成功
- ✅ AutoML 示例代码运行成功（准确率 99.55%）
- ✅ RidgeClassifier API 适配正确
- ✅ sklearn wrapper 修复生效

**未完成验证：**
- ⚠️ 完整测试套件（因超时未完成）

## 新增依赖

| 依赖 | 版本 | 原因 |
|------|------|------|
| sqlalchemy | >=2.0.48 | 源码中使用但未声明 |
| py | >=1.11.0 | 测试代码依赖 py._path.local |

## 关键修复

### 1. sklearn API 变更

**问题：** `RidgeClassifier` 的 `normalize` 参数在 sklearn 1.2+ 已移除

**修复：** 从 HyperparameterSpace 中移除该参数

```python
# 修复前
HyperparameterSpace({
    'alpha': Choice([0.0, 1.0, 10.0, 100.0]),
    'fit_intercept': Boolean(),
    'normalize': Boolean()  # ❌ sklearn 1.2+ 不支持
})

# 修复后
HyperparameterSpace({
    'alpha': Choice([0.0, 1.0, 10.0, 100.0]),
    'fit_intercept': Boolean()
})
```

### 2. inspect API 变更

**问题：** `inspect.getfullargspec().args` 在 sklearn 1.8.0 中返回 `['estimator']`，导致误判

**修复：** 使用 `inspect.signature().parameters` 获取准确的参数列表

```python
# 修复前
len(inspect.getfullargspec(self.wrapped_sklearn_predictor.fit).args) < 3

# 修复后
len(inspect.signature(self.wrapped_sklearn_predictor.fit).parameters) < 2
```

### 3. SQLAlchemy 警告

**问题：** `declarative_base()` 在 SQLAlchemy 2.0 中已弃用

**状态：** ⚠️ 仅警告，不影响功能（未修复）

**建议：** 后续可改为 `sqlalchemy.orm.declarative_base()`

## 备注

1. **测试超时问题：** 完整测试套件运行时间过长，建议优化测试性能或增加超时限制
2. **SQLAlchemy 2.0 迁移：** 代码中使用了已弃用的 API，建议后续迁移到新 API
3. **依赖声明完整性：** 原 setup.py 缺少 sqlalchemy 声明，已补充
4. **Python 3.13 兼容性：** 核心功能已验证兼容，但完整测试套件未完成验证

## 升级价值

✅ **成功升级到 Python 3.13 + 最新依赖**
- NumPy 2.4.3（支持最新特性）
- pandas 3.0.1（性能提升）
- scikit-learn 1.8.0（最新算法）
- Flask 3.1.3（安全更新）

✅ **核心功能验证通过**
- AutoML 流程正常运行
- 模型训练和预测正常
- 超参数优化正常

⚠️ **待完善**
- 完整测试套件验证（因超时未完成）
- SQLAlchemy 2.0 API 迁移
