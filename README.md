# Subpoena · 传票

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/fisher0131/citation-subpoena/blob/main/LICENSE)

**Make every citation testify.** 给一篇写完的报告，逐条给每个引用发传票，让被引来源自己出庭作证：它到底支不支持它绑定的那句话。

## 它解决什么

报告里的 `[1]` 不等于事实。一条引用可以真实存在、来源真实可读，却对被绑定的陈述**沉默**——这就是引用幻觉。Subpoena 不检验事实真值，只检验**引用绑定**：把陈述拆开、抓来源正文、逐字核对，然后出判决。

## 安装

```powershell
pip install subpoena
```

## 用法

```powershell
# 离线审计：不联网、不要 API、零成本（语料库 jsonl 每行一条 {url, text} 快照）
subpoena corpus/demo/report.md --corpus corpus/demo/sources.jsonl --output results/audit.json

# 真实抓取被引来源正文（HTML/PDF/XML），失败自动分类，学者来源被墙走开放获取回退
subpoena report.md --citations citations.json --fetch live

# 换成 LLM 核验（默认 rule 模式是确定性字符串匹配，够准且免费）
subpoena report.md --citations citations.json --corpus sources.jsonl \
    --verify model --model gpt-4o-mini --api-key sk-...
```

引用标记解析支持 markdown 脚注 `[^1]`、行内 `[1]`、`(Author, Year)` 和 `[text](url)`。报告自带 `[^1]: url` 脚注定义行时，`--citations` 可省。

`--fail-on-hallucination 0.2` 可当 CI 门禁：幻觉率超阈值即返回退出码 1。

## 判决标签

| 标签 | 含义 |
|---|---|
| `SUPPORTED` | 被引来源逐字支持该陈述 |
| `G1_CITATION_BINDING` | 陈述有别的来源支持，但绑错了来源（引用沉默） |
| `G2_EVIDENCE_GAP` | 证据池里没有任何来源支持 |
| `G3_CONTRADICTED` | 来源明确给出不相容的值 |
| `FETCH_FAILURE` | 来源抓不到——**不得**计为幻觉，也不得计为正确 |

幻觉率 = (G1+G2+G3) / 可判定陈述数，附 Wilson 95% 置信区间。

## Fail-closed 原则

所有判定默认不信任：模型输出格式不对、quote 不在来源快照里逐字出现、或来源抓取失败，一律降级为 `UNVERIFIABLE` 或 `FETCH_FAILURE`，绝不默认成 `SUPPORTED`。

## 局限

- 只检验引用绑定（grounding），不检验事实真值：`G2` 只说明"证据池里没有"，不等于该陈述为假。
- 离线语料模式下，语料覆盖度直接决定结果——陈述在给定语料内定位不到即 `G2`。

## API 从哪来

`--verify model` 需要你自己提供**任意 OpenAI 兼容**的 chat completions 端点：`--api-key`（或环境变量 `SUBPOENA_API_KEY`）+ `--model`，`--api-base-url` 可指向 OpenAI / DeepSeek / 本地 vLLM 等。框架内不内置任何 key。默认 `--verify rule` 模式完全不需要 API。

## 许可证

MIT。
