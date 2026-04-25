"use client";

import { useState, type CSSProperties, type PointerEvent } from "react";
import Image, { type StaticImageData } from "next/image";
import styles from "./ComparisonSlider.module.css";

type ComparisonSliderProps = {
  afterAlt: string;
  afterLabel: string;
  afterSrc: StaticImageData;
  beforeAlt: string;
  beforeLabel: string;
  beforeSrc: StaticImageData;
};

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

export function ComparisonSlider({
  afterAlt,
  afterLabel,
  afterSrc,
  beforeAlt,
  beforeLabel,
  beforeSrc,
}: ComparisonSliderProps) {
  const [position, setPosition] = useState(0.54);

  function updatePosition(event: PointerEvent<HTMLDivElement>) {
    const bounds = event.currentTarget.getBoundingClientRect();
    const next = clamp((event.clientX - bounds.left) / bounds.width, 0, 1);
    setPosition(next);
  }

  return (
    <figure className={styles.figure}>
      <div
        aria-label={`${beforeLabel} and ${afterLabel} comparison slider`}
        className={styles.slider}
        onDragStart={(event) => event.preventDefault()}
        onPointerDown={(event) => {
          event.preventDefault();
          event.currentTarget.setPointerCapture(event.pointerId);
          updatePosition(event);
        }}
        onPointerMove={(event) => {
          if (event.buttons !== 1) {
            return;
          }

          updatePosition(event);
        }}
        onPointerLeave={() => setPosition(0.54)}
        onPointerUp={(event) => {
          updatePosition(event);
          event.currentTarget.releasePointerCapture(event.pointerId);
        }}
        role="img"
        style={{ "--position": `${position * 100}%` } as CSSProperties}
      >
        <Image alt={beforeAlt} className={styles.image} draggable={false} priority src={beforeSrc} />

        <div className={styles.overlay}>
          <Image alt={afterAlt} className={styles.image} draggable={false} src={afterSrc} />
        </div>

        <div aria-hidden className={styles.divider}>
          <span className={styles.handle} />
        </div>

        <div className={styles.badges}>
          <span className={styles.badge}>{beforeLabel}</span>
          <span className={styles.badge}>{afterLabel}</span>
        </div>
      </div>
    </figure>
  );
}
