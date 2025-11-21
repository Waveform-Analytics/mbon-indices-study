'use client';

import React, { useMemo } from 'react';
import * as d3 from 'd3';
import BaseHeatmap, { ProcessedDataPoint } from './BaseHeatmap';
import { useHeatmapData } from './hooks/useHeatmapData';
import { MIDNIGHT_CENTERED_Y_AXIS } from './configs/axisConfigs';

interface RmsSplHeatmapProps {
  className?: string;
  useMidnightCentering?: boolean;
}

// SPL bandwidth configuration
const SPL_BANDWIDTHS = [
  { key: 'Broadband (1-40000 Hz)', label: 'Broadband' },
  { key: 'Low (50-1200 Hz)', label: 'Low Frequency' },
  { key: 'High (7000-40000 Hz)', label: 'High Frequency' }
];

const RmsSplHeatmap: React.FC<RmsSplHeatmapProps> = ({
  className = '',
  useMidnightCentering = true
}) => {
  const {
    loading,
    error,
    stations,
    selectedStation,
    selectedVariable: selectedBandwidth,
    setSelectedStation,
    setSelectedVariable: setSelectedBandwidth,
    filteredData
  } = useHeatmapData({
    dataUrl: '/views/02_environmental_aligned_2021.json', // Same file as environmental
    excludeKeys: ['datetime', 'station', 'year'],
    defaultStation: '9M',
    defaultVariable: 'Broadband (1-40000 Hz)'
  });

  // Get current bandwidth configuration
  const currentBandwidthConfig = SPL_BANDWIDTHS.find(b => b.key === selectedBandwidth) ||
                                 SPL_BANDWIDTHS[0];

  // Process data for heatmap
  const processedData = useMemo<ProcessedDataPoint[]>(() => {
    if (!filteredData.length || !selectedBandwidth) return [];

    const processed: ProcessedDataPoint[] = [];

    filteredData.forEach(d => {
      const value = d[selectedBandwidth] as number;
      // Only add data points with valid values
      if (value !== null && value !== undefined && !isNaN(value)) {
        const date = new Date(d.datetime);
        const dayOfYear = Math.floor(
          (date.getTime() - new Date(Date.UTC(date.getUTCFullYear(), 0, 0)).getTime()) / 86400000
        );

        processed.push({
          day: dayOfYear,
          hour: date.getUTCHours(),
          value: value,
          date: date
        });
      }
    });

    return processed;
  }, [filteredData, selectedBandwidth]);

  // Custom tooltip formatter for SPL data
  const formatTooltip = (d: ProcessedDataPoint) => {
    return `Date: ${d.date.toLocaleDateString()}<br/>
            Hour: ${d.originalHour || d.hour}:00<br/>
            ${currentBandwidthConfig.label}: ${d.value.toFixed(2)} dB`;
  };

  return (
    <div className={className}>
      {/* Selection controls */}
      <div className="flex gap-4 mb-4">
        <div>
          <label className="block text-sm font-medium mb-1">Station</label>
          <select
            className="px-3 py-2 border border-gray-200 rounded-md text-sm bg-background"
            value={selectedStation}
            onChange={(e) => setSelectedStation(e.target.value)}
          >
            {stations.map(station => (
              <option key={station} value={station}>{station}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Bandwidth</label>
          <select
            className="px-3 py-2 border border-gray-200 rounded-md text-sm bg-background"
            value={selectedBandwidth}
            onChange={(e) => setSelectedBandwidth(e.target.value)}
          >
            {SPL_BANDWIDTHS.map(bandwidth => (
              <option key={bandwidth.key} value={bandwidth.key}>{bandwidth.label}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Base heatmap */}
      <BaseHeatmap
        data={processedData}
        loading={loading}
        error={error}
        colorScheme="plasma"
        formatTooltip={formatTooltip}
        legendLabel={`${currentBandwidthConfig.label} SPL (dB)`}
        legendFormatter={(v) => d3.format('.1f')(v)}
        yAxisConfig={useMidnightCentering ? MIDNIGHT_CENTERED_Y_AXIS : undefined}
        noDataMessage={`No SPL data available for station ${selectedStation}`}
      />
    </div>
  );
};

export default RmsSplHeatmap;