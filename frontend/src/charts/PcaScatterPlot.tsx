import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";
import type { PcaPoint } from "@/types/api";

interface PcaScatterPlotProps {
  points: PcaPoint[];
  explainedVariance?: number[];
  height?: number;
}

const CATEGORY_STYLE: Record<
  PcaPoint["category"],
  { color: string; size: number; opacity: number; symbol: string }
> = {
  "Unlabeled Pool": {
    color: "#A89A8F",
    size: 6.5,
    opacity: 0.45,
    symbol: "circle",
  },
  "Already Labeled": {
    color: "#B98255",
    size: 8,
    opacity: 0.9,
    symbol: "circle",
  },
  "Newly Queried by Active Learning": {
    color: "#6B3E26",
    size: 11,
    opacity: 1.0,
    symbol: "diamond",
  },
};

export function PcaScatterPlot({
  points,
  explainedVariance = [0, 0],
  height = 420,
}: PcaScatterPlotProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el || !points.length) return;

    const categories: PcaPoint["category"][] = [
      "Unlabeled Pool",
      "Already Labeled",
      "Newly Queried by Active Learning",
    ];

    const traces = categories
      .map((cat) => {
        const subset = points.filter((p) => p.category === cat);
        if (!subset.length) return null;
        const style = CATEGORY_STYLE[cat];
        return {
          type: "scatter",
          mode: "markers",
          name: `${cat} (${subset.length})`,
          x: subset.map((p) => p.pc1),
          y: subset.map((p) => p.pc2),
          text: subset.map((p) => p.sample_id),
          hovertemplate:
            "<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra>" +
            cat +
            "</extra>",
          marker: {
            color: style.color,
            size: style.size,
            opacity: style.opacity,
            symbol: style.symbol,
            line: {
              color: "#FFFFFF",
              width: cat === "Newly Queried by Active Learning" ? 1.5 : 0.8,
            },
          },
        };
      })
      .filter(Boolean) as Record<string, unknown>[];

    const pc1Var = ((explainedVariance[0] || 0) * 100).toFixed(1);
    const pc2Var = ((explainedVariance[1] || 0) * 100).toFixed(1);

    Plotly.react(
      el,
      traces,
      {
        height,
        margin: { l: 48, r: 16, t: 12, b: 64 },
        paper_bgcolor: "#FFFCF7",
        plot_bgcolor: "#F6EFE7",
        font: { family: "Inter, sans-serif", size: 12, color: "#38271F" },
        xaxis: {
          title: `PC1 (${pc1Var}% var)`,
          gridcolor: "#E6D6C6",
          zerolinecolor: "#D4BFAC",
        },
        yaxis: {
          title: `PC2 (${pc2Var}% var)`,
          gridcolor: "#E6D6C6",
          zerolinecolor: "#D4BFAC",
        },
        legend: {
          orientation: "h",
          y: -0.22,
          x: 0.5,
          xanchor: "center",
          font: { size: 11, color: "#38271F" },
        },
        hovermode: "closest",
      },
      {
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
      }
    );

    return () => {
      if (el) Plotly.purge(el);
    };
  }, [points, explainedVariance, height]);

  if (!points.length) {
    return (
      <div className="flex h-[360px] items-center justify-center rounded-xl border border-dashed border-[#E6D6C6] bg-[#FFFCF7] text-xs text-[#806D60]">
        2D PCA latent embedding coordinates will render once a model round finishes training.
      </div>
    );
  }

  return <div ref={containerRef} className="w-full" />;
}

