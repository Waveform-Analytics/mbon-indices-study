/**
 * Export all heatmap components
 */

export { default as BaseHeatmap } from './BaseHeatmap';
export { default as DetectionsHeatmap } from './DetectionsHeatmap';
export { default as AcousticIndicesHeatmap } from './AcousticIndicesHeatmap';
export { default as RmsSplHeatmap } from './RmsSplHeatmap';
export { default as EnvironmentalHeatmap } from './EnvironmentalHeatmap';

// Export types
export type { ProcessedDataPoint } from './BaseHeatmap';

// Export hooks and configs
export { useHeatmapData } from './hooks/useHeatmapData';
export { STANDARD_Y_AXIS, MIDNIGHT_CENTERED_Y_AXIS } from './configs/axisConfigs';
export type { YAxisConfig } from './configs/axisConfigs';