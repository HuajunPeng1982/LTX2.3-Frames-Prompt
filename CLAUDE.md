# LTX2.3 Frames Prompt — 开发经验总结

## ComfyUI 节点开发要点

### 1. 返回值格式
- `OUTPUT_NODE = True` 时，必须返回 `{"ui": {"name": [value]}, "result": (value,)}` 格式
- `ui` 里的数据才会渲染到节点的 UI widget 上
- 普通元组 `(value,)` 只作为输出端口数据，不会在节点内显示

### 2. Widget 配置
- `forceInput: True` 会把文本输入框变成只能连线的端口，导致 UI 布局大乱，慎用
- ComfyUI 版本间的 widget 签名差异（如 `IS_CHANGED` 警告）通常无害，不必强行修复

### 3. JS 前端
- 自定义 JS 前端可能和 ComfyUI 原生 OUTPUT_NODE 行为冲突
- 优先用原生机制，不行再加 JS
- `WEB_DIRECTORY` 必须是相对于 `__init__.py` 的目录名字符串，不是绝对路径

### 4. 输出策略选择
- `OUTPUT_NODE = True`：结果直接显示在节点内，适合简单输出
- 标准输出端口：用户自行连线到 ShowText 节点，兼容性最好
- **最佳实践**：用 `OUTPUT_NODE = True` + `ui` 字典返回

### 5. 图片输入
- 图片用 `image_1`（required） + `image_2..image_N`（optional）模式
- ComfyUI IMAGE tensor 格式：(B, H, W, C) float32 [0,1]
- 取单张用 `tensor[0]` 降到 (H, W, C)

## API 通信踩坑

### 1. SDK vs 直连
- `google-genai` SDK 对第三方 API 代理兼容性差
- 第三方代理用 OpenAI 兼容格式 + `requests` 直连更可靠
- 用 `requests` 可以看到完整请求/响应，方便调试

### 2. 结构化输出
- `response_mime_type` + `response_schema` 很多第三方 API 不支持
- 改为 prompt 内嵌 JSON 格式指令，配合手动解析更兼容
- 解析时做多层降级：structured → JSON → markdown code fence 剥离 → 原始文本

### 3. 网络问题
- 302 重定向会让 POST 变 GET，请求体丢失 → 用正确 URL，禁用重定向跟随
- 服务器可能有 HTTP_PROXY 环境变量 → 加 `proxies={"http": None, "https": None}`
- 429 限流需要专门处理，等待时间要比普通重试长（10-15秒）

### 4. 超时设置
- 多图片 + 大模型生成需要较长超时，6 分钟（360秒）是合理值
- `requests` 的 timeout 单位是秒，google-genai SDK 的 timeout 单位是毫秒

## 调试流程

### 关键日志点
- `[开始]` — 确认节点被调用
- `[输入]` — 图片数量和帧对数，确认输入解析正确
- `[模型]` `[地址]` — 确认参数传递正确
- `[请求]` — 确认 API 端点和超时配置
- `[响应]` — HTTP 状态码和响应长度
- `[调试]` — 原始响应内容前 N 字符，定位 API 返回格式
- `[完成]` `[失败]` — 最终状态

### 排查步骤
1. 先看 status 日志确认到哪一步挂了
2. 网络问题：检查 URL、代理、超时
3. 响应问题：看 [调试] 输出的原始内容
4. 解析问题：加降级逻辑，用原始文本兜底

## 部署注意
- 服务器代码更新后必须重启 ComfyUI
- `pip install` 确认用对 Python 环境（conda vs system vs embed）
- 本地改完代码 → 手动上传服务器 → 验证 → 重启，这个循环要严格执行
