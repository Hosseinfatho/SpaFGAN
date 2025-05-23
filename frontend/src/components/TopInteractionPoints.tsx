import React, { useState, useEffect } from 'react';
import { Button, Select, Space, Typography, message } from 'antd';
import { LeftOutlined, RightOutlined, CalculatorOutlined, DownloadOutlined } from '@ant-design/icons';

const { Option } = Select;
const { Text } = Typography;

interface Point {
  x: number;
  y: number;
  value: number;
  z?: number;
  zoom?: number;
}

interface TopInteractionPointsProps {
  groupId: number;
  onPointSelect: (point: Point) => void;
  onGroupChange: (groupId: number) => void;
}

const TopInteractionPoints: React.FC<TopInteractionPointsProps> = ({ 
  groupId, 
  onPointSelect,
  onGroupChange 
}) => {
  const [points, setPoints] = useState<Point[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [numPoints, setNumPoints] = useState(10);
  const [calculatingAll, setCalculatingAll] = useState(false);

  const groupOptions = [
    { value: 1, label: 'Endothelial-immune interface (CD31 + CD11b)' },
    { value: 2, label: 'ROS detox, immune stress (CD11b + Catalase)' },
    { value: 3, label: 'T/B cell recruitment via vessels (CD31 + CD4/CD20)' },
    { value: 4, label: 'T–B collaboration (CD4 + CD20)' }
  ];

  const fetchTopPoints = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/top_interaction_points', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          group_id: groupId,
          num_points: numPoints,
        }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch top points');
      }
      
      const data = await response.json();
      if (data.points && data.points.length > 0) {
        setPoints(data.points);
        setCurrentIndex(0);
        message.success('Points loaded successfully');
      } else {
        message.warning('No points found for this group');
      }
    } catch (error) {
      console.error('Error fetching top points:', error);
      message.error('Failed to fetch top points');
    } finally {
      setLoading(false);
    }
  };

  const calculateAllGroups = async () => {
    setCalculatingAll(true);
    try {
      const response = await fetch('http://localhost:5000/api/calculate_all_top_points', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          num_points: 20,
        }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to calculate all groups');
      }
      
      const data = await response.json();
      if (data.status === 'success') {
        message.success('Successfully calculated and saved top points for all groups');
        await fetchTopPoints();
      } else {
        throw new Error(data.message);
      }
    } catch (error) {
      console.error('Error calculating all groups:', error);
      message.error('Failed to calculate all groups');
    } finally {
      setCalculatingAll(false);
    }
  };

  useEffect(() => {
    fetchTopPoints();
  }, [groupId, numPoints]);

  const handlePrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
    }
  };

  const handleNext = () => {
    if (currentIndex < points.length - 1) {
      setCurrentIndex(currentIndex + 1);
    }
  };

  const handleSetPoint = () => {
    if (points[currentIndex]) {
      onPointSelect({
        ...points[currentIndex],
        z: 0,
        zoom: -1.1
      });
    }
  };

  const handleNumPointsChange = (value: number) => {
    setNumPoints(value);
  };

  const handleGroupChange = (value: number) => {
    onGroupChange(value);
  };

  return (
    <div style={{ 
      position: 'absolute',
      right: '200px',
      top: '10px',
      padding: '8px',
      background: 'rgba(255, 255, 255, 0.5)',
      borderRadius: '4px',
      boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)',
      width: '250px',
      zIndex: 1000
    }}>
      <Space direction="vertical" size="small" style={{ width: '100%' }}>
        <Space direction="vertical" size="small" style={{ width: '100%' }}>
          <Text strong style={{ fontSize: '11px' }}>Group:</Text>
          <Select
            value={groupId}
            onChange={handleGroupChange}
            size="small"
            style={{ width: '100%' }}
          >
            {groupOptions.map(option => (
              <Option key={option.value} value={option.value}>
                {option.label}
              </Option>
            ))}
          </Select>
        </Space>

        <Space style={{ width: '100%', justifyContent: 'space-between' }}>
          <Button 
            type="primary"
            size="small"
            icon={<CalculatorOutlined />}
            onClick={fetchTopPoints}
            loading={loading}
            style={{ fontSize: '11px' }}
          >
            Current
          </Button>
          
          <Button 
            type="primary"
            size="small"
            icon={<DownloadOutlined />}
            onClick={calculateAllGroups}
            loading={calculatingAll}
            style={{ fontSize: '11px' }}
          >
            All
          </Button>
        </Space>

        <Space size="small">
          <Text strong style={{ fontSize: '11px' }}>Points:</Text>
          <Select 
            value={numPoints} 
            onChange={handleNumPointsChange}
            size="small"
            style={{ width: '75px' }}
          >
            <Option value={5}>5</Option>
            <Option value={10}>10</Option>
            <Option value={20}>20</Option>
          </Select>

          <Button 
            type="primary"
            size="small"
            icon={<LeftOutlined />}
            onClick={handlePrevious}
            disabled={currentIndex === 0}
            style={{ fontSize: '11px' }}
          />
          <Button 
            type="primary"
            size="small"
            icon={<RightOutlined />}
            onClick={handleNext}
            disabled={currentIndex === points.length - 1}
            style={{ fontSize: '11px' }}
          />
        </Space>

        {points[currentIndex] && (
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            padding: '2px 4px',
            background: 'rgba(0, 0, 0, 0.05)',
            borderRadius: '4px',
            fontSize: '9px',
            gap: '2px'
          }}>
            <Text style={{ flex: 1, fontSize: '9px' }}>X:{points[currentIndex].x}</Text>
            <Text style={{ flex: 1, fontSize: '9px' }}>Y:{points[currentIndex].y}</Text>
            <Text style={{ flex: 1, fontSize: '9px' }}>V:{points[currentIndex].value.toFixed(2)}</Text>
            <Button 
              type="primary"
              onClick={handleSetPoint}
              disabled={!points[currentIndex]}
              size="small"
              style={{ flex: 1, fontSize: '9px', padding: '0 4px', height: '20px' }}
            >
              Set
            </Button>
          </div>
        )}

      </Space>
    </div>
  );
};

export default TopInteractionPoints; 