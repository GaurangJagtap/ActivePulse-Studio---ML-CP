import { useCallback, useEffect, useState } from "react";
import { ErrorBanner, LoadingState } from "@/components/shared/FeedbackBanner";
import { DashboardLayout } from "@/layouts/DashboardLayout";
import { ActiveLearningLabPage } from "@/pages/ActiveLearningLabPage";
import { AnnotationStudioPage } from "@/pages/AnnotationStudioPage";
import { BenchmarkArenaPage } from "@/pages/BenchmarkArenaPage";
import { DatasetExplorerPage } from "@/pages/DatasetExplorerPage";
import { ExperimentHistoryPage } from "@/pages/ExperimentHistoryPage";
import { OverviewDashboardPage } from "@/pages/OverviewDashboardPage";
import { api } from "@/services/api";
import type {
  DatasetSummary,
  ExperimentDetailResponse,
  ExperimentSummary,
  HealthResponse,
  ModelInfo,
  NavSectionId,
  StrategyInfo,
} from "@/types/api";

export function App() {
  const [section, setSection] = useState<NavSectionId>(() => {
    const saved = sessionStorage.getItem("activepulse_last_section");
    if (saved && ["overview", "datasets", "lab", "annotation", "benchmarks", "history"].includes(saved)) {
      return saved as NavSectionId;
    }
    return "overview";
  });
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [strategies, setStrategies] = useState<StrategyInfo[]>([]);
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [selectedExpId, setSelectedExpId] = useState<string | null>(null);
  const [activeDetail, setActiveDetail] =
    useState<ExperimentDetailResponse | null>(null);

  const [bootLoading, setBootLoading] = useState(true);
  const [bootError, setBootError] = useState<string | null>(null);

  const loadExperimentDetail = useCallback(async (id: string) => {
    try {
      const detail = await api.getExperiment(id);
      setActiveDetail(detail);
    } catch {
      setActiveDetail(null);
    }
  }, []);

  const refreshAll = useCallback(async () => {
    setBootError(null);
    try {
      const [h, ds, mods, strats, exps] = await Promise.all([
        api.health(),
        api.listDatasets(),
        api.listModels(),
        api.listStrategies(),
        api.listExperiments(),
      ]);
      setHealth(h);
      setDatasets(ds);
      setModels(mods);
      setStrategies(strats);
      setExperiments(exps);

      const targetId =
        selectedExpId && exps.some((e) => e.id === selectedExpId)
          ? selectedExpId
          : exps[0]?.id || null;

      setSelectedExpId(targetId);
      if (targetId) {
        await loadExperimentDetail(targetId);
      } else {
        setActiveDetail(null);
      }
    } catch (err) {
      setHealth(null);
      setBootError(
        err instanceof Error
          ? `Unable to reach ActivePulse FastAPI backend: ${err.message}`
          : "Unable to reach ActivePulse FastAPI backend."
      );
    } finally {
      setBootLoading(false);
    }
  }, [selectedExpId, loadExperimentDetail]);

  useEffect(() => {
    refreshAll();
  }, [refreshAll]);

  const handleSelectExperiment = useCallback(
    async (id: string) => {
      setSelectedExpId(id);
      await loadExperimentDetail(id);
    },
    [loadExperimentDetail]
  );

  const handleExperimentCreated = useCallback(
    async (id: string) => {
      setSelectedExpId(id);
      const exps = await api.listExperiments();
      setExperiments(exps);
      await loadExperimentDetail(id);
    },
    [loadExperimentDetail]
  );

  const handleRefreshExperiment = useCallback(
    async (id: string) => {
      const [exps] = await Promise.all([
        api.listExperiments(),
        loadExperimentDetail(id),
      ]);
      setExperiments(exps);
    },
    [loadExperimentDetail]
  );

  return (
    <DashboardLayout
      activeSection={section}
      onSelectSection={setSection}
      experiments={experiments}
      selectedExperimentId={selectedExpId}
      onSelectExperiment={handleSelectExperiment}
      health={health}
      onRefreshAll={refreshAll}
    >
      {bootError && <ErrorBanner message={bootError} className="mb-6" />}

      {bootLoading ? (
        <LoadingState label="Connecting to ActivePulse Studio ML backend..." />
      ) : (
        <>
          {section === "overview" && (
            <OverviewDashboardPage
              experiments={experiments}
              activeDetail={activeDetail}
              onNavigate={setSection}
              onSelectExperiment={handleSelectExperiment}
            />
          )}

          {section === "datasets" && (
            <DatasetExplorerPage
              datasets={datasets}
              experiments={experiments}
              selectedExperimentId={selectedExpId}
            />
          )}

          {section === "lab" && (
            <ActiveLearningLabPage
              datasets={datasets}
              models={models}
              strategies={strategies}
              selectedExperimentId={selectedExpId}
              activeDetail={activeDetail}
              onExperimentCreated={handleExperimentCreated}
              onRefreshExperiment={handleRefreshExperiment}
              onNavigate={setSection}
            />
          )}

          {section === "annotation" && (
            <AnnotationStudioPage
              datasets={datasets}
              experiments={experiments}
              selectedExperimentId={selectedExpId}
              onSelectExperiment={handleSelectExperiment}
              onExperimentCreated={handleExperimentCreated}
              onRefreshExperiment={handleRefreshExperiment}
            />
          )}

          {section === "benchmarks" && (
            <BenchmarkArenaPage datasets={datasets} strategies={strategies} />
          )}

          {section === "history" && (
            <ExperimentHistoryPage
              experiments={experiments}
              selectedExperimentId={selectedExpId}
              onSelectExperiment={handleSelectExperiment}
              onRefreshList={refreshAll}
              onNavigate={setSection}
            />
          )}
        </>
      )}
    </DashboardLayout>
  );
}

export default App;

