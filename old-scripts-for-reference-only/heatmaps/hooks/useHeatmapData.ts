/**
 * Custom hook for fetching and managing heatmap data
 */

import { useState, useEffect, useRef } from 'react';

export interface HeatmapDataPoint {
  datetime: number;
  station: string;
  year: number;
  [key: string]: string | number;
}

export interface UseHeatmapDataConfig {
  // URL path relative to CDN base (e.g., '/views/02_detections_aligned_2021.json')
  dataUrl: string;
  // Keys to exclude when extracting variable names
  excludeKeys: string[];
  // Default selections
  defaultStation?: string;
  defaultVariable?: string;
  // Optional data transformer
  transformData?: (data: HeatmapDataPoint[]) => HeatmapDataPoint[];
}

export interface UseHeatmapDataReturn {
  // Raw data
  data: HeatmapDataPoint[];
  loading: boolean;
  error: string | null;
  // Extracted metadata
  stations: string[];
  variables: string[];
  // Selection state
  selectedStation: string;
  selectedVariable: string;
  setSelectedStation: (station: string) => void;
  setSelectedVariable: (variable: string) => void;
  // Filtered data for current selection
  filteredData: HeatmapDataPoint[];
}

/**
 * Hook for fetching and managing heatmap data from CDN
 */
export const useHeatmapData = (config: UseHeatmapDataConfig): UseHeatmapDataReturn => {
  const { dataUrl, defaultStation = '9M', defaultVariable = '' } = config;

  // Use refs to store stable values
  const configRef = useRef(config);
  configRef.current = config;

  const [data, setData] = useState<HeatmapDataPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [stations, setStations] = useState<string[]>([]);
  const [variables, setVariables] = useState<string[]>([]);
  const [selectedStation, setSelectedStation] = useState<string>(defaultStation);
  const [selectedVariable, setSelectedVariable] = useState<string>(defaultVariable);

  // Fetch data on mount
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        const cdnUrl = process.env.NEXT_PUBLIC_CDN_BASE_URL || '';
        const response = await fetch(`${cdnUrl}${dataUrl}`);

        if (!response.ok) {
          throw new Error(`Failed to fetch data: ${response.statusText}`);
        }

        let jsonData: HeatmapDataPoint[] = await response.json();

        // Apply optional data transformation
        const currentConfig = configRef.current;
        if (currentConfig.transformData) {
          jsonData = currentConfig.transformData(jsonData);
        }

        setData(jsonData);

        // Extract unique stations
        const uniqueStations = Array.from(new Set(jsonData.map(d => d.station))).sort();
        setStations(uniqueStations);

        // Extract variable columns (excluding metadata fields)
        if (jsonData.length > 0) {
          const allKeys = Object.keys(jsonData[0]);
          console.log(`[useHeatmapData] All keys in ${dataUrl}:`, allKeys);

          const variableKeys = allKeys
            .filter(key => !currentConfig.excludeKeys.includes(key))
            .filter(key => typeof jsonData[0][key] === 'number' ||
                          (typeof jsonData[0][key] === 'string' && !['station'].includes(key)))
            .sort();
          setVariables(variableKeys);
          console.log(`[useHeatmapData] Filtered variables for ${dataUrl}:`, variableKeys);

          // Set default variable if not already set
          if (!currentConfig.defaultVariable && variableKeys.length > 0) {
            setSelectedVariable(variableKeys[0]);
          }
        }

        setLoading(false);
      } catch (err) {
        console.error('Error fetching heatmap data:', err);
        setError(err instanceof Error ? err.message : 'Failed to load data');
        setLoading(false);
      }
    };

    fetchData();
  }, [dataUrl, defaultVariable]); // Remove array/function dependencies that change on every render

  // Filter data based on current selection
  const filteredData = data.filter(d => d.station === selectedStation);

  return {
    data,
    loading,
    error,
    stations,
    variables,
    selectedStation,
    selectedVariable,
    setSelectedStation,
    setSelectedVariable,
    filteredData,
  };
};