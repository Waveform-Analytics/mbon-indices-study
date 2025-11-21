'use client';

import React, { useMemo, useState, useEffect } from 'react';
import * as d3 from 'd3';
import BaseHeatmap, { ProcessedDataPoint } from './BaseHeatmap';
import { useHeatmapData } from './hooks/useHeatmapData';
import { MIDNIGHT_CENTERED_Y_AXIS } from './configs/axisConfigs';

interface ClusterMetadata {
  index_name: string;
  cluster_id: number;
  cluster_size: number;
  is_selected: boolean;
  selection_rationale: string;
}

interface AcousticIndicesHeatmapProps {
  className?: string;
  useMidnightCentering?: boolean;
}

const AcousticIndicesHeatmap: React.FC<AcousticIndicesHeatmapProps> = ({
  className = '',
  useMidnightCentering = true
}) => {
  const [clusterMetadata, setClusterMetadata] = useState<ClusterMetadata[]>([]);
  const [clusters, setClusters] = useState<string[]>([]);
  const [selectedCluster, setSelectedCluster] = useState<string>('All');
  const {
    loading,
    error,
    stations,
    variables: allIndices,
    selectedStation,
    selectedVariable: selectedIndex,
    setSelectedStation,
    setSelectedVariable: setSelectedIndex,
    filteredData
  } = useHeatmapData({
    dataUrl: '/views/acoustic_indices_full.json',
    excludeKeys: ['datetime', 'station', 'year', 'FrequencyResolution', 'hour'],
    defaultStation: '9M',
    defaultVariable: 'ACTspFract'
  });

  // Fetch cluster metadata
  useEffect(() => {
    const fetchClusterMetadata = async () => {
      try {
        const cdnUrl = process.env.NEXT_PUBLIC_CDN_BASE_URL || '';
        const response = await fetch(`${cdnUrl}/views/acoustic_indices_clusters.json`);

        if (response.ok) {
          const metadata: ClusterMetadata[] = await response.json();
          setClusterMetadata(metadata);

          // Extract unique clusters
          const uniqueClusters = Array.from(new Set(metadata.map(m => m.cluster_id)))
            .sort((a, b) => a - b)
            .map(id => `Cluster ${id}`);
          setClusters(['All', ...uniqueClusters]);
        }
      } catch (err) {
        console.error('Error fetching cluster metadata:', err);
        // Continue without cluster metadata
      }
    };

    fetchClusterMetadata();
  }, []);

  // Get available indices based on cluster selection
  const availableIndices = useMemo(() => {
    if (selectedCluster === 'All') {
      return allIndices;
    }

    if (!clusterMetadata.length) {
      return allIndices;
    }

    const clusterId = parseInt(selectedCluster.replace('Cluster ', ''));
    return clusterMetadata
      .filter(m => m.cluster_id === clusterId)
      .map(m => m.index_name)
      .filter(name => allIndices.includes(name))
      .sort();
  }, [clusterMetadata, selectedCluster, allIndices]);

  // Auto-select representative index when cluster changes
  useEffect(() => {
    if (availableIndices.length > 0 && !availableIndices.includes(selectedIndex)) {
      // Find representative index for the cluster, or fall back to first available
      const representative = clusterMetadata
        .filter(m => availableIndices.includes(m.index_name))
        .find(m => m.is_selected);

      setSelectedIndex(representative?.index_name || availableIndices[0]);
    }
  }, [availableIndices, selectedIndex, clusterMetadata, setSelectedIndex]);

  // Process data for heatmap
  const processedData = useMemo<ProcessedDataPoint[]>(() => {
    if (!filteredData.length || !selectedIndex) return [];

    return filteredData.map(d => {
      const date = new Date(d.datetime);
      const dayOfYear = Math.floor(
        (date.getTime() - new Date(Date.UTC(date.getUTCFullYear(), 0, 0)).getTime()) / 86400000
      );

      return {
        day: dayOfYear,
        hour: date.getUTCHours(),
        value: d[selectedIndex] as number,
        date: date
      };
    });
  }, [filteredData, selectedIndex]);

  // Custom tooltip formatter for acoustic indices
  const formatTooltip = (d: ProcessedDataPoint) => {
    return `Date: ${d.date.toLocaleDateString()}<br/>
            Hour: ${d.originalHour || d.hour}:00<br/>
            ${selectedIndex}: ${d.value.toFixed(4)}`;
  };

  return (
    <div className={className}>
      {/* Selection controls */}
      <div className="flex gap-4 mb-4 flex-wrap">
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

        {clusters.length > 0 && (
          <div>
            <label className="block text-sm font-medium mb-1">Cluster Group</label>
            <select
              className="px-3 py-2 border border-gray-200 rounded-md text-sm bg-background"
              value={selectedCluster}
              onChange={(e) => setSelectedCluster(e.target.value)}
            >
              {clusters.map(cluster => (
                <option key={cluster} value={cluster}>{cluster}</option>
              ))}
            </select>
          </div>
        )}

        <div>
          <label className="block text-sm font-medium mb-1">Acoustic Index</label>
          <select
            className="px-3 py-2 border border-gray-200 rounded-md text-sm bg-background"
            value={selectedIndex}
            onChange={(e) => setSelectedIndex(e.target.value)}
          >
            {availableIndices.map(index => {
              const metadata = clusterMetadata.find(m => m.index_name === index);
              const isRepresentative = metadata?.is_selected || false;
              return (
                <option key={index} value={index}>
                  {index} {isRepresentative ? '★' : ''}
                </option>
              );
            })}
          </select>
        </div>
      </div>

      {/* Base heatmap */}
      <BaseHeatmap
        data={processedData}
        loading={loading}
        error={error}
        colorScheme="viridis"
        formatTooltip={formatTooltip}
        legendLabel={selectedIndex}
        legendFormatter={(v) => d3.format('.3f')(v)}
        yAxisConfig={useMidnightCentering ? MIDNIGHT_CENTERED_Y_AXIS : undefined}
        noDataMessage={`No ${selectedIndex} data available for station ${selectedStation}`}
      />
    </div>
  );
};

export default AcousticIndicesHeatmap;