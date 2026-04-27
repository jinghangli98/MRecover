"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import { ComparisonSlider } from "@/components/ComparisonSlider";
import styles from "./page.module.css";
import t1CaseOne from "../src/assets/t1_cor_1.png";
import generatedCaseOne from "../src/assets/t2w_cor_1.png";
import targetCaseOne from "../src/assets/tse_cor_1.png";
import t1CaseTwo from "../src/assets/t1_cor_2.png";
import generatedCaseTwo from "../src/assets/t2w_cor_2.png";
import targetCaseTwo from "../src/assets/tse_cor_2.png";

const huggingFaceModelUrl = "https://huggingface.co/jil202/MRecover";

const cases = {
  en: [
    {
      id: "mprage",
      eyebrow: "Case 01",
      label: "T1w MPRAGE -> TSE",
      note: "Autoregressive synthesis from T1w image to T2w-TSE contrast, validated against the as-acquired reference for downstream ASHS subfield segmentation",
      images: [
        { alt: "MPRAGE source image", label: "Source T1w", src: t1CaseOne },
        { alt: "Translated TSE image for MPRAGE case", label: "Translated TSE", src: generatedCaseOne },
        { alt: "As-acquired TSE image for MPRAGE case", label: "As-acquired TSE", src: targetCaseOne },
      ],
    },
    {
      id: "mp2rage",
      eyebrow: "Case 02",
      label: "T1w MP2RAGE -> TSE",
      note: "",
      images: [
        { alt: "MP2RAGE source image", label: "Source T1w", src: t1CaseTwo },
        { alt: "Translated TSE image for MP2RAGE case", label: "Translated TSE", src: generatedCaseTwo },
        { alt: "As-acquired TSE image for MP2RAGE case", label: "As-acquired TSE", src: targetCaseTwo },
      ],
    },
  ],
  zh: [
    {
      id: "mprage",
      eyebrow: "案例 01",
      label: "T1w MPRAGE -> TSE",
      note: "基于 T1w 结构像合成 T2w-TSE 对比，并与实际采集的 TSE 参考图像对照验证，用于后续 ASHS 海马亚区分割分析。",
      images: [
        { alt: "MPRAGE 原始 T1w 图像", label: "原始 T1w", src: t1CaseOne },
        { alt: "MPRAGE 案例的合成 TSE 图像", label: "合成 TSE", src: generatedCaseOne },
        { alt: "MPRAGE 案例的实采 TSE 参考图像", label: "实采 TSE", src: targetCaseOne },
      ],
    },
    {
      id: "mp2rage",
      eyebrow: "案例 02",
      label: "T1w MP2RAGE -> TSE",
      note: "",
      images: [
        { alt: "MP2RAGE 原始 T1w 图像", label: "原始 T1w", src: t1CaseTwo },
        { alt: "MP2RAGE 案例的合成 TSE 图像", label: "合成 TSE", src: generatedCaseTwo },
        { alt: "MP2RAGE 案例的实采 TSE 参考图像", label: "实采 TSE", src: targetCaseTwo },
      ],
    },
  ],
} as const;

const copy = {
  en: {
    themeSwitchLabel: { light: "Night", dark: "Day" },
    languageButton: "中",
    proofIntro: "Compare synthesized T2w-TSE directly against the as-acquired reference — generated from T1w alone.",
    scientificContext: "Scientific context",
    scientificTitle: "Designed to recover motion corrupted T2w TSE data",
    scientificCopy:
      "High-resolution T2w-TSE is the gold standard for hippocampal subfield analysis, but motion artifacts hit hardest in elderly and cognitively impaired patients. We synthesize it from T1w, which everyone already has.",
    scaleEyebrow: "Built for Scale",
    scaleTitle: "Full volume synthesized in under 30 seconds",
    scaleCopy:
      "Single-step ODE sampling on an NVIDIA A100, no iterative denoising, no prohibitive compute cost. Fast enough for large-scale cohort workflows, practical enough for retrospective datasets where T2w-TSE was never acquired.",
    requestEyebrow: "Request access",
    requestTitle: "Model access is managed on Hugging Face",
    requestCopy:
      "MRecover weights are hosted as a gated Hugging Face model. Request access there with your name, email, affiliation, and intended research use.",
    accessCta: "Open gated model",
    accessNote: "After approval, authenticate with Hugging Face before running the package.",
    accessSteps: ["Open the model page", "Submit the Hugging Face access request", "Run huggingface-cli login"],
  },
  zh: {
    themeSwitchLabel: { light: "夜", dark: "日" },
    languageButton: "En",
    proofIntro: "展示仅基于 T1w 输入合成的 T2w-TSE 图像，并与实际采集的 TSE 参考图像进行逐例对照。",
    scientificContext: "临床背景",
    scientificTitle: "面向受运动伪影影响的 T2w-TSE 重建",
    scientificCopy:
      "高分辨率 T2w-TSE 是海马亚区定量分析的重要成像基础，但在老年受试者及认知功能受损人群中，原始采集往往更易受到运动伪影影响。MRecover 利用常规可得的 T1w 结构像生成对应的 T2w-TSE 对比，以补足下游分析所需信息。",
    scaleEyebrow: "规模化应用",
    scaleTitle: "30 秒内完成整卷推理",
    scaleCopy:
      "在 NVIDIA A100 上可通过单步 ODE 采样于 30 秒内完成整卷生成，无需迭代去噪流程，也无需高额计算开销。该流程适用于大规模队列研究，也适用于历史数据中缺失 T2w-TSE 采集的回顾性分析。",
    requestEyebrow: "访问申请",
    requestTitle: "模型访问由 Hugging Face 统一管理",
    requestCopy:
      "MRecover 权重托管为 Hugging Face gated model。请在模型页面提交姓名、邮箱、机构及科研用途说明以申请访问权限。",
    accessCta: "打开 gated model",
    accessNote: "通过审核后，请先完成 Hugging Face 认证，再运行本工具包。",
    accessSteps: ["打开模型页面", "提交 Hugging Face 访问申请", "运行 huggingface-cli login"],
  },
} as const;

export default function Home() {
  const [activeCase, setActiveCase] = useState<(typeof cases.en)[number]["id"]>("mprage");
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [language, setLanguage] = useState<"en" | "zh">("en");

  const currentCopy = copy[language];
  const localizedCases = cases[language];
  const currentCase = localizedCases.find((item) => item.id === activeCase) ?? localizedCases[0];

  useEffect(() => {
    const storedTheme = window.localStorage.getItem("mrecover-theme");
    const storedLanguage = window.localStorage.getItem("mrecover-language");

    if (storedTheme === "light" || storedTheme === "dark") {
      setTheme(storedTheme);
    } else if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
      setTheme("dark");
    }

    if (storedLanguage === "en" || storedLanguage === "zh") {
      setLanguage(storedLanguage);
    }
  }, []);

  function toggleTheme() {
    const nextTheme = theme === "light" ? "dark" : "light";
    setTheme(nextTheme);
    window.localStorage.setItem("mrecover-theme", nextTheme);
  }

  function toggleLanguage() {
    const nextLanguage = language === "en" ? "zh" : "en";
    setLanguage(nextLanguage);
    window.localStorage.setItem("mrecover-language", nextLanguage);
  }

  return (
    <main className={styles.page} data-language={language} data-theme={theme} lang={language === "zh" ? "zh-CN" : "en"}>
      <div className={styles.backgroundGlow} />
      <div className={styles.controlStack}>
        <button
          aria-label={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
          className={styles.themeSwitch}
          onClick={toggleTheme}
          type="button"
        >
          <span className={styles.themeSwitchTrack}>
            <span className={styles.themeSwitchLabel}>{currentCopy.themeSwitchLabel[theme]}</span>
            <span className={theme === "dark" ? styles.themeSwitchKnobActive : styles.themeSwitchKnob} />
          </span>
        </button>

        <button
          aria-label={language === "en" ? "Switch page language to Chinese" : "Switch page language to English"}
          className={styles.languageSwitch}
          onClick={toggleLanguage}
          type="button"
        >
          {currentCopy.languageButton}
        </button>
      </div>
      <section className={styles.proofSection}>
        <div className={styles.sectionHeader}>
          <p className={styles.kicker}>MRecover</p>
          <p className={styles.sectionCopy}>{currentCopy.proofIntro}</p>
        </div>

        <div className={styles.caseSwitch}>
          {localizedCases.map((item) => (
            <button
              key={item.id}
              className={item.id === activeCase ? styles.caseButtonActive : styles.caseButton}
              onClick={() => setActiveCase(item.id)}
              type="button"
            >
              <span>{item.eyebrow}</span>
              <strong>{item.label}</strong>
            </button>
          ))}
        </div>

        <div className={styles.caseCard}>
          <div className={styles.caseIntro}>
            <div>
              <p className={styles.caseEyebrow}>{currentCase.eyebrow}</p>
              <h3 className={styles.caseTitle}>{currentCase.label}</h3>
            </div>
            <p className={styles.caseNote}>{currentCase.note}</p>
          </div>

          <div className={styles.proofLayout}>
            <figure className={styles.sourceFrame}>
              <div className={styles.imageShell}>
                <Image
                  alt={currentCase.images[0].alt}
                  className={styles.comparisonImage}
                  draggable={false}
                  priority
                  src={currentCase.images[0].src}
                />
              </div>
            </figure>

            <div className={styles.sliderWrap}>
              <ComparisonSlider
                afterAlt={currentCase.images[2].alt}
                afterLabel={currentCase.images[2].label}
                afterSrc={currentCase.images[2].src}
                beforeAlt={currentCase.images[1].alt}
                beforeLabel={currentCase.images[1].label}
                beforeSrc={currentCase.images[1].src}
              />
            </div>
          </div>
        </div>
      </section>

      <section className={styles.detailsSection}>
        <div className={styles.detailsGrid}>
          <article className={styles.detailCard}>
            <p className={styles.sectionEyebrow}>{currentCopy.scientificContext}</p>
            <h3 className={styles.detailTitle}>{currentCopy.scientificTitle}</h3>
            <p className={styles.detailCopy}>{currentCopy.scientificCopy}</p>
          </article>

          <article className={styles.detailCard}>
            <p className={styles.sectionEyebrow}>{currentCopy.scaleEyebrow}</p>
            <h3 className={styles.detailTitle}>{currentCopy.scaleTitle}</h3>
            <p className={styles.detailCopy}>{currentCopy.scaleCopy}</p>
          </article>
        </div>
      </section>

      <section className={styles.contactSection}>
        <div className={styles.contactIntro}>
          <p className={styles.sectionEyebrow}>{currentCopy.requestEyebrow}</p>
          <h2 className={styles.sectionTitle}>{currentCopy.requestTitle}</h2>
          <p className={styles.sectionCopy}>{currentCopy.requestCopy}</p>
        </div>

        <div className={styles.contactCard}>
          <ol className={styles.accessSteps}>
            {currentCopy.accessSteps.map((step) => (
              <li key={step}>{step}</li>
            ))}
          </ol>
          <a className={styles.accessButton} href={huggingFaceModelUrl} rel="noreferrer" target="_blank">
            {currentCopy.accessCta}
          </a>
          <p className={styles.accessNote}>{currentCopy.accessNote}</p>
        </div>
      </section>
    </main>
  );
}
