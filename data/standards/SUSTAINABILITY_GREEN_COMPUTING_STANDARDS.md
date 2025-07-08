# Sustainability and Green Computing Standards

## 1. Overview

This document establishes standards for sustainable software development and green computing practices to minimize environmental impact while maintaining system performance and reliability.

### Purpose

- Define carbon footprint assessment methodologies for software systems
- Establish energy-efficient coding and architecture practices
- Guide sustainable cloud computing and infrastructure decisions
- Promote circular economy principles in technology lifecycle management

### Scope

These standards apply to:
- Software development lifecycle processes
- Cloud infrastructure and deployment patterns
- Hardware procurement and lifecycle management
- Data center operations and optimization
- DevOps and CI/CD pipeline efficiency

## 2. Carbon Footprint Assessment

### 2.1 Software Carbon Intensity Measurement

**Standard**: Implement comprehensive carbon footprint tracking for software systems

```python
# Example: Carbon footprint assessment framework
import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import requests
import psutil

@dataclass
class CarbonFootprint:
    timestamp: datetime
    cpu_usage_kwh: float
    memory_usage_kwh: float
    storage_usage_kwh: float
    network_usage_kwh: float
    total_kwh: float
    carbon_intensity_g_kwh: float
    total_co2_grams: float

class CarbonTracker:
    def __init__(self, region='US-WEST-1'):
        self.region = region
        self.baseline_measurements = {}
        self.carbon_coefficients = {
            'cpu': 0.0003,      # kWh per CPU hour at 100%
            'memory': 0.000375, # kWh per GB hour
            'storage_ssd': 0.0000017, # kWh per GB hour
            'storage_hdd': 0.0000004, # kWh per GB hour
            'network': 0.000006 # kWh per GB transferred
        }
    
    async def measure_carbon_footprint(self, duration_minutes=1):
        """Measure carbon footprint over specified duration"""
        start_time = time.time()
        start_metrics = self._get_system_metrics()
        
        # Wait for measurement period
        await asyncio.sleep(duration_minutes * 60)
        
        end_metrics = self._get_system_metrics()
        duration_hours = (time.time() - start_time) / 3600
        
        # Calculate energy consumption
        cpu_kwh = self._calculate_cpu_energy(
            start_metrics['cpu_percent'],
            end_metrics['cpu_percent'],
            duration_hours
        )
        
        memory_kwh = self._calculate_memory_energy(
            start_metrics['memory_gb'],
            duration_hours
        )
        
        storage_kwh = self._calculate_storage_energy(
            start_metrics['disk_gb'],
            duration_hours
        )
        
        network_kwh = self._calculate_network_energy(
            start_metrics['network_gb'],
            end_metrics['network_gb']
        )
        
        total_kwh = cpu_kwh + memory_kwh + storage_kwh + network_kwh
        
        # Get carbon intensity for region
        carbon_intensity = await self._get_carbon_intensity()
        total_co2_grams = total_kwh * carbon_intensity
        
        return CarbonFootprint(
            timestamp=datetime.now(),
            cpu_usage_kwh=cpu_kwh,
            memory_usage_kwh=memory_kwh,
            storage_usage_kwh=storage_kwh,
            network_usage_kwh=network_kwh,
            total_kwh=total_kwh,
            carbon_intensity_g_kwh=carbon_intensity,
            total_co2_grams=total_co2_grams
        )
    
    def _get_system_metrics(self) -> Dict:
        """Get current system resource metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        return {
            'cpu_percent': cpu_percent,
            'memory_gb': memory.used / (1024**3),
            'disk_gb': disk.used / (1024**3),
            'network_gb': (network.bytes_sent + network.bytes_recv) / (1024**3)
        }
    
    def _calculate_cpu_energy(self, start_cpu, end_cpu, duration_hours):
        """Calculate CPU energy consumption"""
        avg_cpu_percent = (start_cpu + end_cpu) / 2
        cpu_cores = psutil.cpu_count()
        return (avg_cpu_percent / 100) * cpu_cores * duration_hours * self.carbon_coefficients['cpu']
    
    async def _get_carbon_intensity(self) -> float:
        """Get current carbon intensity for region"""
        try:
            # Example API call to carbon intensity service
            response = requests.get(
                f"https://api.carbonintensity.org.uk/regional/regionid/{self.region}",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                return data['data'][0]['intensity']['actual']
            else:
                # Fallback to regional average
                return self._get_regional_carbon_average()
        except Exception:
            return self._get_regional_carbon_average()
    
    def _get_regional_carbon_average(self) -> float:
        """Get regional carbon intensity averages"""
        regional_averages = {
            'US-WEST-1': 350,    # California - lower due to renewables
            'US-EAST-1': 450,    # Virginia
            'EU-WEST-1': 300,    # Ireland - high wind power
            'EU-CENTRAL-1': 400, # Germany
            'AP-SOUTHEAST-1': 550, # Singapore
            'AP-NORTHEAST-1': 500  # Japan
        }
        return regional_averages.get(self.region, 450)
    
    def generate_carbon_report(self, measurements: List[CarbonFootprint]) -> Dict:
        """Generate comprehensive carbon footprint report"""
        if not measurements:
            return {}
        
        total_co2 = sum(m.total_co2_grams for m in measurements)
        total_kwh = sum(m.total_kwh for m in measurements)
        
        avg_carbon_intensity = sum(m.carbon_intensity_g_kwh for m in measurements) / len(measurements)
        
        # Calculate breakdown by component
        cpu_co2 = sum(m.cpu_usage_kwh * m.carbon_intensity_g_kwh for m in measurements)
        memory_co2 = sum(m.memory_usage_kwh * m.carbon_intensity_g_kwh for m in measurements)
        storage_co2 = sum(m.storage_usage_kwh * m.carbon_intensity_g_kwh for m in measurements)
        network_co2 = sum(m.network_usage_kwh * m.carbon_intensity_g_kwh for m in measurements)
        
        return {
            'summary': {
                'total_co2_grams': total_co2,
                'total_kwh': total_kwh,
                'avg_carbon_intensity': avg_carbon_intensity,
                'measurement_period': {
                    'start': min(m.timestamp for m in measurements),
                    'end': max(m.timestamp for m in measurements),
                    'duration_hours': len(measurements)
                }
            },
            'breakdown': {
                'cpu_co2_grams': cpu_co2,
                'memory_co2_grams': memory_co2,
                'storage_co2_grams': storage_co2,
                'network_co2_grams': network_co2,
                'percentages': {
                    'cpu': (cpu_co2 / total_co2) * 100,
                    'memory': (memory_co2 / total_co2) * 100,
                    'storage': (storage_co2 / total_co2) * 100,
                    'network': (network_co2 / total_co2) * 100
                }
            },
            'recommendations': self._generate_optimization_recommendations(measurements)
        }
    
    def _generate_optimization_recommendations(self, measurements: List[CarbonFootprint]) -> List[str]:
        """Generate recommendations based on carbon footprint analysis"""
        recommendations = []
        
        avg_cpu_kwh = sum(m.cpu_usage_kwh for m in measurements) / len(measurements)
        avg_memory_kwh = sum(m.memory_usage_kwh for m in measurements) / len(measurements)
        
        if avg_cpu_kwh > 0.1:  # High CPU usage
            recommendations.append(
                "Consider CPU optimization: profile code for inefficient algorithms and implement caching"
            )
        
        if avg_memory_kwh > 0.05:  # High memory usage
            recommendations.append(
                "Optimize memory usage: implement memory pooling and reduce object allocations"
            )
        
        # Regional recommendations
        if self.region in ['US-EAST-1', 'AP-SOUTHEAST-1']:
            recommendations.append(
                "Consider migrating to regions with cleaner energy sources (e.g., US-WEST-1, EU-WEST-1)"
            )
        
        return recommendations

# Usage example
async def main():
    tracker = CarbonTracker(region='US-WEST-1')
    
    # Measure carbon footprint during application load
    measurements = []
    for i in range(10):  # 10 minute measurement
        footprint = await tracker.measure_carbon_footprint(duration_minutes=1)
        measurements.append(footprint)
        print(f"Measurement {i+1}: {footprint.total_co2_grams:.2f}g CO2")
    
    # Generate report
    report = tracker.generate_carbon_report(measurements)
    print(f"Total CO2 emissions: {report['summary']['total_co2_grams']:.2f}g")
    print("Optimization recommendations:")
    for rec in report['recommendations']:
        print(f"- {rec}")
```

**NIST Mappings**: SI-12 (information handling), AU-6 (audit review)

### 2.2 Infrastructure Carbon Assessment

**Standard**: Implement cloud infrastructure carbon tracking

```python
# Example: Cloud infrastructure carbon assessment
import boto3
from datetime import datetime, timedelta
from typing import Dict, List

class CloudCarbonAssessment:
    def __init__(self, cloud_provider='aws'):
        self.cloud_provider = cloud_provider
        self.carbon_factors = {
            'aws': {
                'us-west-1': 0.350,      # kg CO2e per kWh
                'us-west-2': 0.285,
                'us-east-1': 0.447,
                'eu-west-1': 0.316,
                'ap-southeast-1': 0.431
            },
            'azure': {
                'westus': 0.350,
                'westus2': 0.285,
                'eastus': 0.447,
                'westeurope': 0.316
            },
            'gcp': {
                'us-west1': 0.350,
                'us-central1': 0.381,
                'europe-west1': 0.316,
                'asia-southeast1': 0.431
            }
        }
    
    def assess_compute_footprint(self, instances: List[Dict]) -> Dict:
        """Assess carbon footprint of compute instances"""
        total_co2 = 0
        total_cost = 0
        instance_breakdown = []
        
        for instance in instances:
            # Get instance specifications
            vcpus = instance.get('vcpus', 1)
            memory_gb = instance.get('memory_gb', 1)
            runtime_hours = instance.get('runtime_hours', 24)
            region = instance.get('region', 'us-east-1')
            
            # Calculate power consumption
            cpu_power_watts = vcpus * 3.5  # Approximate CPU power per vCPU
            memory_power_watts = memory_gb * 0.5  # Approximate memory power per GB
            total_power_watts = cpu_power_watts + memory_power_watts
            
            # Add infrastructure overhead (cooling, networking, etc.)
            pue = 1.4  # Power Usage Effectiveness
            total_power_with_overhead = total_power_watts * pue
            
            # Calculate energy consumption
            energy_kwh = (total_power_with_overhead / 1000) * runtime_hours
            
            # Calculate carbon emissions
            carbon_factor = self.carbon_factors[self.cloud_provider].get(region, 0.450)
            co2_kg = energy_kwh * carbon_factor
            
            instance_breakdown.append({
                'instance_id': instance.get('id', 'unknown'),
                'instance_type': instance.get('type', 'unknown'),
                'region': region,
                'energy_kwh': energy_kwh,
                'co2_kg': co2_kg,
                'cost_usd': instance.get('cost_usd', 0)
            })
            
            total_co2 += co2_kg
            total_cost += instance.get('cost_usd', 0)
        
        return {
            'total_co2_kg': total_co2,
            'total_cost_usd': total_cost,
            'carbon_efficiency': total_co2 / total_cost if total_cost > 0 else 0,
            'instances': instance_breakdown,
            'recommendations': self._generate_compute_recommendations(instance_breakdown)
        }
    
    def assess_storage_footprint(self, storage_configs: List[Dict]) -> Dict:
        """Assess carbon footprint of storage systems"""
        total_co2 = 0
        storage_breakdown = []
        
        storage_carbon_factors = {
            'ssd': 0.65,    # kg CO2e per TB per year
            'hdd': 0.32,    # kg CO2e per TB per year
            'cold': 0.15    # kg CO2e per TB per year
        }
        
        for storage in storage_configs:
            storage_type = storage.get('type', 'ssd')
            size_tb = storage.get('size_gb', 0) / 1024
            duration_days = storage.get('duration_days', 365)
            region = storage.get('region', 'us-east-1')
            
            # Calculate annual carbon footprint
            annual_co2 = size_tb * storage_carbon_factors.get(storage_type, 0.65)
            
            # Prorate for actual duration
            co2_kg = annual_co2 * (duration_days / 365)
            
            # Apply regional carbon intensity
            carbon_factor = self.carbon_factors[self.cloud_provider].get(region, 0.450)
            regional_multiplier = carbon_factor / 0.450  # Normalize to baseline
            co2_kg *= regional_multiplier
            
            storage_breakdown.append({
                'storage_id': storage.get('id', 'unknown'),
                'type': storage_type,
                'size_tb': size_tb,
                'region': region,
                'co2_kg': co2_kg,
                'cost_usd': storage.get('cost_usd', 0)
            })
            
            total_co2 += co2_kg
        
        return {
            'total_co2_kg': total_co2,
            'storage_breakdown': storage_breakdown,
            'recommendations': self._generate_storage_recommendations(storage_breakdown)
        }
    
    def _generate_compute_recommendations(self, instances: List[Dict]) -> List[str]:
        """Generate compute optimization recommendations"""
        recommendations = []
        
        # Check for oversized instances
        high_carbon_instances = [i for i in instances if i['co2_kg'] > 100]
        if high_carbon_instances:
            recommendations.append(
                f"Consider rightsizing {len(high_carbon_instances)} high-carbon instances"
            )
        
        # Check for inefficient regions
        high_carbon_regions = set(
            i['region'] for i in instances 
            if self.carbon_factors[self.cloud_provider].get(i['region'], 0.450) > 0.400
        )
        if high_carbon_regions:
            recommendations.append(
                f"Consider migrating from high-carbon regions: {', '.join(high_carbon_regions)}"
            )
        
        # Check carbon efficiency
        inefficient_instances = [
            i for i in instances 
            if i['cost_usd'] > 0 and (i['co2_kg'] / i['cost_usd']) > 0.1
        ]
        if inefficient_instances:
            recommendations.append(
                "Some instances have poor carbon efficiency - consider ARM-based instances"
            )
        
        return recommendations
    
    def create_sustainability_dashboard(self, assessment_data: Dict) -> Dict:
        """Create sustainability metrics dashboard"""
        return {
            'overview': {
                'total_co2_kg_month': assessment_data.get('total_co2_kg', 0) * 30,
                'co2_equivalent': {
                    'trees_to_offset': (assessment_data.get('total_co2_kg', 0) * 30) / 21.77,
                    'car_miles_equivalent': (assessment_data.get('total_co2_kg', 0) * 30) / 0.404
                },
                'carbon_intensity_trend': self._calculate_carbon_trend(),
                'efficiency_score': self._calculate_efficiency_score(assessment_data)
            },
            'targets': {
                'carbon_reduction_goal': '30% by 2025',
                'renewable_energy_target': '100% by 2030',
                'current_renewable_percentage': self._get_renewable_percentage()
            },
            'actions': {
                'immediate': [
                    'Implement auto-shutdown for dev environments',
                    'Migrate to ARM-based instances',
                    'Enable intelligent tiering for storage'
                ],
                'short_term': [
                    'Migrate to low-carbon regions',
                    'Implement carbon-aware scheduling',
                    'Optimize application efficiency'
                ],
                'long_term': [
                    'Achieve carbon neutrality',
                    'Implement circular economy practices',
                    'Zero-waste data center operations'
                ]
            }
        }
```

**NIST Mappings**: SI-4 (information system monitoring), AU-2 (audit events)

## 3. Energy-Efficient Coding Practices

### 3.1 Algorithmic Efficiency

**Standard**: Optimize algorithms for energy efficiency alongside performance

```python
# Example: Energy-efficient algorithm implementations
import time
import psutil
from functools import wraps
from typing import Callable, Any, Dict

def energy_profile(func: Callable) -> Callable:
    """Decorator to profile energy consumption of functions"""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Measure before execution
        start_time = time.time()
        start_cpu = psutil.cpu_percent(interval=0.1)
        start_memory = psutil.virtual_memory().used
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Measure after execution
        end_time = time.time()
        end_cpu = psutil.cpu_percent(interval=0.1)
        end_memory = psutil.virtual_memory().used
        
        # Calculate energy metrics
        execution_time = end_time - start_time
        avg_cpu = (start_cpu + end_cpu) / 2
        memory_delta = end_memory - start_memory
        
        # Estimate energy consumption
        cpu_energy = (avg_cpu / 100) * execution_time * 3.5  # Watts per core
        memory_energy = (memory_delta / (1024**3)) * execution_time * 0.5  # Watts per GB
        total_energy = cpu_energy + memory_energy
        
        # Store metrics
        if not hasattr(func, '_energy_metrics'):
            func._energy_metrics = []
        
        func._energy_metrics.append({
            'execution_time': execution_time,
            'cpu_percentage': avg_cpu,
            'memory_delta_mb': memory_delta / (1024**2),
            'estimated_energy_wh': total_energy / 3600,
            'timestamp': time.time()
        })
        
        return result
    
    return wrapper

class EnergyEfficientAlgorithms:
    """Collection of energy-optimized algorithm implementations"""
    
    @staticmethod
    @energy_profile
    def efficient_sort(data: list, algorithm='timsort') -> list:
        """Energy-efficient sorting with algorithm selection"""
        if len(data) < 100:
            # For small datasets, simple algorithms are more energy efficient
            return sorted(data)
        elif algorithm == 'timsort':
            # Python's built-in sort is highly optimized
            return sorted(data)
        elif algorithm == 'quicksort':
            # Implement in-place quicksort for memory efficiency
            return EnergyEfficientAlgorithms._quicksort_inplace(data.copy(), 0, len(data) - 1)
        else:
            return sorted(data)
    
    @staticmethod
    def _quicksort_inplace(arr: list, low: int, high: int) -> list:
        """In-place quicksort to minimize memory allocation"""
        if low < high:
            pi = EnergyEfficientAlgorithms._partition(arr, low, high)
            EnergyEfficientAlgorithms._quicksort_inplace(arr, low, pi - 1)
            EnergyEfficientAlgorithms._quicksort_inplace(arr, pi + 1, high)
        return arr
    
    @staticmethod
    def _partition(arr: list, low: int, high: int) -> int:
        """Partition function for quicksort"""
        pivot = arr[high]
        i = low - 1
        
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1
    
    @staticmethod
    @energy_profile
    def efficient_search(data: list, target: Any, algorithm='binary') -> int:
        """Energy-efficient search with algorithm selection"""
        if not data:
            return -1
        
        if algorithm == 'binary' and len(data) > 100:
            # Binary search for large sorted datasets
            return EnergyEfficientAlgorithms._binary_search(data, target, 0, len(data) - 1)
        else:
            # Linear search for small datasets or unsorted data
            try:
                return data.index(target)
            except ValueError:
                return -1
    
    @staticmethod
    def _binary_search(arr: list, target: Any, low: int, high: int) -> int:
        """Binary search implementation"""
        if low <= high:
            mid = (low + high) // 2
            
            if arr[mid] == target:
                return mid
            elif arr[mid] > target:
                return EnergyEfficientAlgorithms._binary_search(arr, target, low, mid - 1)
            else:
                return EnergyEfficientAlgorithms._binary_search(arr, target, mid + 1, high)
        
        return -1
    
    @staticmethod
    @energy_profile
    def efficient_matrix_multiply(a: list, b: list) -> list:
        """Energy-efficient matrix multiplication"""
        rows_a, cols_a = len(a), len(a[0])
        rows_b, cols_b = len(b), len(b[0])
        
        if cols_a != rows_b:
            raise ValueError("Matrix dimensions incompatible")
        
        # Use blocked algorithm for cache efficiency
        if rows_a > 64 and cols_b > 64:
            return EnergyEfficientAlgorithms._blocked_matrix_multiply(a, b)
        else:
            # Simple algorithm for small matrices
            result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
            
            for i in range(rows_a):
                for j in range(cols_b):
                    for k in range(cols_a):
                        result[i][j] += a[i][k] * b[k][j]
            
            return result
    
    @staticmethod
    def _blocked_matrix_multiply(a: list, b: list, block_size: int = 64) -> list:
        """Blocked matrix multiplication for cache efficiency"""
        rows_a, cols_a = len(a), len(a[0])
        rows_b, cols_b = len(b), len(b[0])
        
        result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
        
        for i in range(0, rows_a, block_size):
            for j in range(0, cols_b, block_size):
                for k in range(0, cols_a, block_size):
                    # Multiply blocks
                    for ii in range(i, min(i + block_size, rows_a)):
                        for jj in range(j, min(j + block_size, cols_b)):
                            for kk in range(k, min(k + block_size, cols_a)):
                                result[ii][jj] += a[ii][kk] * b[kk][jj]
        
        return result

# Memory pool for reducing allocations
class MemoryPool:
    """Memory pool to reduce garbage collection overhead"""
    
    def __init__(self, object_type: type, initial_size: int = 100):
        self.object_type = object_type
        self.pool = []
        self.in_use = set()
        
        # Pre-allocate objects
        for _ in range(initial_size):
            self.pool.append(object_type())
    
    def acquire(self):
        """Get an object from the pool"""
        if self.pool:
            obj = self.pool.pop()
        else:
            obj = self.object_type()
        
        self.in_use.add(id(obj))
        return obj
    
    def release(self, obj):
        """Return an object to the pool"""
        obj_id = id(obj)
        if obj_id in self.in_use:
            self.in_use.remove(obj_id)
            # Reset object state if needed
            if hasattr(obj, 'reset'):
                obj.reset()
            self.pool.append(obj)

# Energy-aware data structures
class EnergyEfficientCache:
    """Cache with energy-aware eviction policies"""
    
    def __init__(self, max_size: int = 1000, energy_weight: float = 0.3):
        self.max_size = max_size
        self.energy_weight = energy_weight
        self.cache = {}
        self.access_count = {}
        self.energy_cost = {}
        self.last_access = {}
    
    def get(self, key: str) -> Any:
        """Get value with energy tracking"""
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            self.last_access[key] = time.time()
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any, computation_energy: float = 1.0):
        """Put value with energy cost tracking"""
        if len(self.cache) >= self.max_size:
            self._evict_energy_aware()
        
        self.cache[key] = value
        self.access_count[key] = 1
        self.energy_cost[key] = computation_energy
        self.last_access[key] = time.time()
    
    def _evict_energy_aware(self):
        """Evict based on energy efficiency"""
        if not self.cache:
            return
        
        current_time = time.time()
        scores = {}
        
        for key in self.cache:
            # Calculate energy efficiency score
            access_freq = self.access_count[key]
            time_since_access = current_time - self.last_access[key]
            energy_cost = self.energy_cost[key]
            
            # Higher score means more valuable to keep
            score = (access_freq / (time_since_access + 1)) * (1 / (energy_cost + 0.1))
            scores[key] = score
        
        # Evict lowest scoring item
        evict_key = min(scores.keys(), key=lambda k: scores[k])
        del self.cache[evict_key]
        del self.access_count[evict_key]
        del self.energy_cost[evict_key]
        del self.last_access[evict_key]
```

**NIST Mappings**: SI-2 (flaw remediation), SA-15 (development process)

### 3.2 Resource Optimization

**Standard**: Implement systematic resource optimization strategies

```python
# Example: Comprehensive resource optimization framework
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Generator, Iterator
import gc
import weakref

class ResourceOptimizer:
    """Framework for optimizing resource usage"""
    
    def __init__(self):
        self.optimization_strategies = {
            'memory': self._optimize_memory,
            'cpu': self._optimize_cpu,
            'io': self._optimize_io,
            'network': self._optimize_network
        }
    
    def lazy_loading_iterator(self, data_source: callable, batch_size: int = 1000) -> Generator:
        """Lazy loading iterator to reduce memory footprint"""
        offset = 0
        while True:
            batch = data_source(offset, batch_size)
            if not batch:
                break
            
            for item in batch:
                yield item
            
            offset += batch_size
            
            # Force garbage collection between batches
            if offset % (batch_size * 10) == 0:
                gc.collect()
    
    def streaming_processor(self, data_stream: Iterator, processor: callable, 
                          buffer_size: int = 100) -> Generator:
        """Stream processing with controlled memory usage"""
        buffer = []
        
        for item in data_stream:
            buffer.append(item)
            
            if len(buffer) >= buffer_size:
                # Process buffer
                results = processor(buffer)
                for result in results:
                    yield result
                
                # Clear buffer and collect garbage
                buffer.clear()
                gc.collect()
        
        # Process remaining items
        if buffer:
            results = processor(buffer)
            for result in results:
                yield result
    
    def _optimize_memory(self, context: dict) -> dict:
        """Memory optimization strategies"""
        optimizations = []
        
        # Object pooling recommendation
        if context.get('frequent_allocations', False):
            optimizations.append({
                'strategy': 'object_pooling',
                'description': 'Implement object pooling for frequently allocated objects',
                'potential_savings': '20-40% memory allocation overhead'
            })
        
        # Weak references for caches
        if context.get('large_caches', False):
            optimizations.append({
                'strategy': 'weak_references',
                'description': 'Use weak references for large caches to allow garbage collection',
                'potential_savings': '10-30% memory usage'
            })
        
        # Data structure optimization
        if context.get('large_datasets', False):
            optimizations.append({
                'strategy': 'efficient_data_structures',
                'description': 'Use memory-efficient data structures (arrays, slots)',
                'potential_savings': '30-50% memory usage'
            })
        
        return {
            'resource_type': 'memory',
            'optimizations': optimizations,
            'estimated_impact': self._calculate_optimization_impact(optimizations)
        }
    
    def _optimize_cpu(self, context: dict) -> dict:
        """CPU optimization strategies"""
        optimizations = []
        
        # Vectorization
        if context.get('numeric_processing', False):
            optimizations.append({
                'strategy': 'vectorization',
                'description': 'Use NumPy/vectorized operations for numeric processing',
                'potential_savings': '50-90% CPU cycles'
            })
        
        # Caching
        if context.get('repeated_calculations', False):
            optimizations.append({
                'strategy': 'memoization',
                'description': 'Cache results of expensive calculations',
                'potential_savings': '60-95% CPU for repeated operations'
            })
        
        # Parallelization
        if context.get('parallelizable_tasks', False):
            optimizations.append({
                'strategy': 'parallel_processing',
                'description': 'Use multiprocessing for CPU-intensive tasks',
                'potential_savings': '2x-8x speedup depending on cores'
            })
        
        return {
            'resource_type': 'cpu',
            'optimizations': optimizations,
            'estimated_impact': self._calculate_optimization_impact(optimizations)
        }
    
    def adaptive_resource_management(self, workload_pattern: str) -> dict:
        """Adaptive resource management based on workload patterns"""
        strategies = {}
        
        if workload_pattern == 'batch_processing':
            strategies = {
                'memory': 'streaming_with_batching',
                'cpu': 'parallel_batch_processing',
                'scheduling': 'off_peak_execution',
                'storage': 'temporary_high_performance'
            }
        elif workload_pattern == 'real_time':
            strategies = {
                'memory': 'pre_allocated_pools',
                'cpu': 'priority_scheduling',
                'caching': 'aggressive_caching',
                'latency': 'locality_optimization'
            }
        elif workload_pattern == 'periodic':
            strategies = {
                'resources': 'auto_scaling',
                'storage': 'tiered_storage',
                'scheduling': 'predictive_scaling',
                'optimization': 'scheduled_maintenance'
            }
        
        return {
            'workload_pattern': workload_pattern,
            'strategies': strategies,
            'implementation': self._generate_implementation_plan(strategies)
        }

# Green coding patterns
class GreenCodingPatterns:
    """Collection of green coding patterns and practices"""
    
    @staticmethod
    def energy_efficient_loops():
        """Examples of energy-efficient loop patterns"""
        return {
            'loop_optimization': {
                'bad_example': '''
                # Inefficient: repeated calculations in loop
                for i in range(len(data)):
                    if expensive_function(data[i]) > threshold:
                        process(data[i])
                ''',
                'good_example': '''
                # Efficient: cache expensive calculations
                cache = {}
                for item in data:
                    key = get_cache_key(item)
                    if key not in cache:
                        cache[key] = expensive_function(item)
                    
                    if cache[key] > threshold:
                        process(item)
                '''
            },
            'early_termination': {
                'bad_example': '''
                # Inefficient: always processes all items
                results = []
                for item in large_dataset:
                    result = process(item)
                    if result.is_valid():
                        results.append(result)
                ''',
                'good_example': '''
                # Efficient: early termination with generator
                def process_until_limit(dataset, limit=100):
                    count = 0
                    for item in dataset:
                        result = process(item)
                        if result.is_valid():
                            yield result
                            count += 1
                            if count >= limit:
                                break
                '''
            }
        }
    
    @staticmethod
    def efficient_data_access():
        """Patterns for efficient data access"""
        return {
            'database_optimization': {
                'connection_pooling': '''
                # Use connection pooling
                from sqlalchemy import create_engine
                from sqlalchemy.pool import QueuePool
                
                engine = create_engine(
                    'postgresql://...',
                    poolclass=QueuePool,
                    pool_size=10,
                    max_overflow=20,
                    pool_pre_ping=True
                )
                ''',
                'batch_operations': '''
                # Batch database operations
                def bulk_insert(records):
                    with engine.begin() as conn:
                        conn.execute(
                            table.insert(),
                            records  # Insert multiple records at once
                        )
                '''
            },
            'caching_strategies': {
                'multi_level_cache': '''
                # Multi-level caching
                from functools import lru_cache
                import redis
                
                redis_client = redis.Redis()
                
                @lru_cache(maxsize=1000)  # L1: Memory cache
                def get_expensive_data(key):
                    # L2: Redis cache
                    cached = redis_client.get(f"cache:{key}")
                    if cached:
                        return json.loads(cached)
                    
                    # L3: Database/computation
                    data = expensive_computation(key)
                    redis_client.setex(
                        f"cache:{key}", 
                        3600,  # 1 hour TTL
                        json.dumps(data)
                    )
                    return data
                '''
            }
        }
```

**NIST Mappings**: SA-11 (developer testing), SI-7 (software integrity)

## 4. Green Cloud Computing Patterns

### 4.1 Carbon-Aware Scheduling

**Standard**: Implement carbon-aware scheduling for cloud workloads

```python
# Example: Carbon-aware scheduling system
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests
from dataclasses import dataclass

@dataclass
class WorkloadRequest:
    id: str
    cpu_hours: float
    memory_gb_hours: float
    deadline: datetime
    priority: int  # 1-5, where 1 is highest
    region_preferences: List[str]
    carbon_sensitivity: float  # 0-1, where 1 is most sensitive

class CarbonAwareScheduler:
    """Scheduler that optimizes for both performance and carbon footprint"""
    
    def __init__(self):
        self.carbon_intensity_cache = {}
        self.renewable_energy_schedules = {}
        self.workload_queue = []
        
    async def schedule_workload(self, workload: WorkloadRequest) -> Dict:
        """Schedule workload based on carbon awareness"""
        # Get carbon intensity forecasts for all preferred regions
        carbon_forecasts = await self._get_carbon_forecasts(
            workload.region_preferences,
            hours_ahead=48
        )
        
        # Find optimal scheduling window
        optimal_schedule = self._find_optimal_window(workload, carbon_forecasts)
        
        # Check if immediate execution is required
        if workload.priority == 1 or workload.deadline <= datetime.now() + timedelta(hours=1):
            # Execute immediately in lowest carbon region
            best_region = min(
                workload.region_preferences,
                key=lambda r: carbon_forecasts[r][0]['carbon_intensity']
            )
            return await self._execute_workload(workload, best_region, datetime.now())
        
        # Schedule for optimal carbon window
        return await self._schedule_for_window(workload, optimal_schedule)
    
    async def _get_carbon_forecasts(self, regions: List[str], hours_ahead: int = 48) -> Dict:
        """Get carbon intensity forecasts for regions"""
        forecasts = {}
        
        for region in regions:
            cache_key = f"{region}_{datetime.now().strftime('%Y%m%d_%H')}"
            
            if cache_key in self.carbon_intensity_cache:
                forecasts[region] = self.carbon_intensity_cache[cache_key]
                continue
            
            try:
                # Example API call to carbon intensity service
                response = await self._fetch_carbon_data(region, hours_ahead)
                forecasts[region] = response
                self.carbon_intensity_cache[cache_key] = response
                
            except Exception as e:
                # Fallback to historical averages
                forecasts[region] = self._get_historical_carbon_average(region, hours_ahead)
        
        return forecasts
    
    def _find_optimal_window(self, workload: WorkloadRequest, forecasts: Dict) -> Dict:
        """Find optimal execution window balancing carbon and constraints"""
        best_score = float('inf')
        best_schedule = None
        
        # Check each hour in the forecast period
        for hour_offset in range(48):
            execution_time = datetime.now() + timedelta(hours=hour_offset)
            
            # Skip if past deadline
            if execution_time + timedelta(hours=workload.cpu_hours) > workload.deadline:
                continue
            
            for region in workload.region_preferences:
                carbon_intensity = forecasts[region][hour_offset]['carbon_intensity']
                renewable_percentage = forecasts[region][hour_offset].get('renewable_percentage', 0)
                
                # Calculate carbon emissions for this window
                total_energy_kwh = (workload.cpu_hours * 3.5) / 1000  # Approximate
                carbon_emissions = total_energy_kwh * carbon_intensity / 1000  # kg CO2
                
                # Calculate scheduling score
                score = self._calculate_schedule_score(
                    workload, carbon_emissions, renewable_percentage, hour_offset
                )
                
                if score < best_score:
                    best_score = score
                    best_schedule = {
                        'region': region,
                        'execution_time': execution_time,
                        'carbon_emissions_kg': carbon_emissions,
                        'renewable_percentage': renewable_percentage,
                        'score': score
                    }
        
        return best_schedule
    
    def _calculate_schedule_score(self, workload: WorkloadRequest, carbon_emissions: float,
                                renewable_percentage: float, hour_delay: int) -> float:
        """Calculate scheduling score balancing multiple factors"""
        # Carbon factor (lower is better)
        carbon_factor = carbon_emissions * workload.carbon_sensitivity
        
        # Delay penalty (higher delay = higher penalty)
        delay_penalty = hour_delay * (6 - workload.priority) * 0.1
        
        # Renewable energy bonus (higher renewable = lower score)
        renewable_bonus = (renewable_percentage / 100) * 0.5
        
        # Urgency factor
        time_to_deadline = (workload.deadline - datetime.now()).total_seconds() / 3600
        urgency_factor = max(0, (48 - time_to_deadline) / 48) * workload.priority * 0.2
        
        total_score = carbon_factor + delay_penalty - renewable_bonus + urgency_factor
        return total_score
    
    async def optimize_multi_region_deployment(self, workloads: List[WorkloadRequest]) -> Dict:
        """Optimize deployment across multiple regions for carbon efficiency"""
        # Get current carbon intensity for all regions
        all_regions = set()
        for workload in workloads:
            all_regions.update(workload.region_preferences)
        
        carbon_data = await self._get_carbon_forecasts(list(all_regions), hours_ahead=1)
        
        # Group workloads by carbon sensitivity
        high_sensitivity = [w for w in workloads if w.carbon_sensitivity > 0.7]
        medium_sensitivity = [w for w in workloads if 0.3 <= w.carbon_sensitivity <= 0.7]
        low_sensitivity = [w for w in workloads if w.carbon_sensitivity < 0.3]
        
        deployment_plan = {
            'immediate_deployment': [],
            'scheduled_deployment': [],
            'carbon_savings': 0,
            'total_emissions': 0
        }
        
        # Deploy high sensitivity workloads in greenest regions
        for workload in high_sensitivity:
            best_region = self._find_greenest_region(workload.region_preferences, carbon_data)
            deployment_plan['immediate_deployment'].append({
                'workload_id': workload.id,
                'region': best_region,
                'carbon_intensity': carbon_data[best_region][0]['carbon_intensity'],
                'reasoning': 'High carbon sensitivity - deployed to greenest available region'
            })
        
        # Schedule medium sensitivity workloads for optimal windows
        for workload in medium_sensitivity:
            optimal_schedule = await self.schedule_workload(workload)
            deployment_plan['scheduled_deployment'].append(optimal_schedule)
        
        # Deploy low sensitivity workloads for cost optimization
        for workload in low_sensitivity:
            # Choose based on cost and performance rather than carbon
            best_region = workload.region_preferences[0]  # Default to first preference
            deployment_plan['immediate_deployment'].append({
                'workload_id': workload.id,
                'region': best_region,
                'carbon_intensity': carbon_data[best_region][0]['carbon_intensity'],
                'reasoning': 'Low carbon sensitivity - optimized for cost/performance'
            })
        
        return deployment_plan
    
    def _find_greenest_region(self, regions: List[str], carbon_data: Dict) -> str:
        """Find region with lowest current carbon intensity"""
        return min(
            regions,
            key=lambda r: carbon_data[r][0]['carbon_intensity']
        )
    
    def create_carbon_budget(self, monthly_limit_kg: float) -> Dict:
        """Create and manage carbon budget for cloud operations"""
        return {
            'budget': {
                'monthly_limit_kg': monthly_limit_kg,
                'daily_limit_kg': monthly_limit_kg / 30,
                'current_month_usage': 0,
                'remaining_budget': monthly_limit_kg
            },
            'tracking': {
                'workloads_deferred': 0,
                'carbon_savings_kg': 0,
                'efficiency_improvements': []
            },
            'policies': {
                'high_carbon_threshold': monthly_limit_kg * 0.1,  # 10% of budget
                'auto_defer_threshold': monthly_limit_kg * 0.9,   # 90% of budget
                'emergency_override': False
            },
            'recommendations': [
                'Implement carbon-aware scheduling for batch workloads',
                'Migrate to regions with higher renewable energy percentage',
                'Optimize application efficiency to reduce resource requirements',
                'Use spot instances during low carbon intensity periods'
            ]
        }

# Green deployment patterns
class GreenDeploymentManager:
    """Manager for environmentally conscious deployment strategies"""
    
    def __init__(self):
        self.deployment_strategies = {
            'blue_green_optimized': self._blue_green_carbon_optimized,
            'canary_efficient': self._canary_energy_efficient,
            'rolling_green': self._rolling_deployment_green
        }
    
    def _blue_green_carbon_optimized(self, deployment_config: Dict) -> Dict:
        """Blue-green deployment optimized for carbon efficiency"""
        return {
            'strategy': 'blue_green_carbon_optimized',
            'phases': [
                {
                    'name': 'carbon_assessment',
                    'description': 'Assess carbon impact of new deployment',
                    'actions': [
                        'Measure baseline carbon footprint',
                        'Estimate new version carbon impact',
                        'Calculate carbon delta'
                    ]
                },
                {
                    'name': 'green_environment_setup',
                    'description': 'Set up green environment in low-carbon region',
                    'actions': [
                        'Select region with highest renewable energy',
                        'Use ARM-based instances for efficiency',
                        'Enable auto-scaling based on carbon intensity'
                    ]
                },
                {
                    'name': 'traffic_migration',
                    'description': 'Migrate traffic during low carbon periods',
                    'actions': [
                        'Wait for optimal carbon window',
                        'Gradual traffic shift (10%, 25%, 50%, 100%)',
                        'Monitor carbon metrics during migration'
                    ]
                },
                {
                    'name': 'blue_environment_shutdown',
                    'description': 'Shutdown old environment to save energy',
                    'actions': [
                        'Graceful shutdown of blue environment',
                        'Archive logs and metrics',
                        'Release resources immediately'
                    ]
                }
            ],
            'carbon_optimization': {
                'estimated_savings': '15-25% during deployment',
                'key_factors': [
                    'Region selection based on renewable energy',
                    'Timing deployment for low carbon windows',
                    'Immediate resource release'
                ]
            }
        }
    
    def generate_green_infrastructure_code(self) -> Dict:
        """Generate infrastructure as code with green practices"""
        return {
            'terraform_green_practices': '''
# Green Terraform practices
resource "aws_instance" "green_compute" {
  # Use ARM-based instances for better energy efficiency
  instance_type = "t4g.medium"  # ARM-based Graviton2
  
  # Select region with high renewable energy
  availability_zone = "us-west-1a"  # California - high renewable energy
  
  # Enable detailed monitoring for optimization
  monitoring = true
  
  # Use gp3 storage for better efficiency
  root_block_device {
    volume_type = "gp3"
    volume_size = 20
    encrypted   = true
    
    # Optimize IOPS and throughput
    iops       = 3000
    throughput = 125
  }
  
  # Shutdown during off-hours
  user_data = <<-EOF
    #!/bin/bash
    # Install carbon-aware shutdown script
    cat > /usr/local/bin/carbon-aware-shutdown.sh << 'SCRIPT'
    #!/bin/bash
    CARBON_API="https://api.carbonintensity.org.uk/intensity"
    THRESHOLD=400
    
    CURRENT_INTENSITY=$(curl -s $CARBON_API | jq '.data[0].intensity.actual')
    
    if [ $CURRENT_INTENSITY -gt $THRESHOLD ]; then
      echo "High carbon intensity detected: $CURRENT_INTENSITY g/kWh"
      echo "Scheduling shutdown for low-carbon period"
      shutdown -h +60  # Shutdown in 1 hour
    fi
    SCRIPT
    
    chmod +x /usr/local/bin/carbon-aware-shutdown.sh
    echo "0 */2 * * * /usr/local/bin/carbon-aware-shutdown.sh" | crontab -
  EOF
  
  tags = {
    Purpose           = "green-computing"
    CarbonOptimized  = "true"
    AutoShutdown     = "enabled"
  }
}

# Auto Scaling Group with carbon awareness
resource "aws_autoscaling_group" "green_asg" {
  name = "green-asg"
  
  vpc_zone_identifier = [
    aws_subnet.green_subnet_1a.id,
    aws_subnet.green_subnet_1c.id
  ]
  
  target_group_arns = [aws_lb_target_group.green_tg.arn]
  
  min_size         = 1
  max_size         = 10
  desired_capacity = 2
  
  # Use mixed instance policy for cost and carbon optimization
  mixed_instances_policy {
    instances_distribution {
      on_demand_percentage                = 20
      spot_allocation_strategy           = "capacity-optimized"
    }
    
    launch_template {
      launch_template_specification {
        launch_template_id = aws_launch_template.green_template.id
        version           = "$Latest"
      }
      
      override {
        instance_type = "t4g.small"
        weighted_capacity = 1
      }
      
      override {
        instance_type = "t4g.medium"
        weighted_capacity = 2
      }
    }
  }
  
  # Carbon-aware scaling policy
  tag {
    key                 = "CarbonAware"
    value              = "true"
    propagate_at_launch = true
  }
}
            ''',
            'kubernetes_green_practices': '''
# Green Kubernetes practices
apiVersion: apps/v1
kind: Deployment
metadata:
  name: green-app
  labels:
    carbon-optimized: "true"
spec:
  replicas: 2
  selector:
    matchLabels:
      app: green-app
  template:
    metadata:
      labels:
        app: green-app
        carbon-optimized: "true"
    spec:
      # Node affinity for ARM-based nodes
      affinity:
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            preference:
              matchExpressions:
              - key: kubernetes.io/arch
                operator: In
                values: ["arm64"]
          - weight: 80
            preference:
              matchExpressions:
              - key: carbon-intensity
                operator: In
                values: ["low"]
      
      containers:
      - name: app
        image: myapp:arm64
        
        # Resource limits for efficiency
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 512Mi
        
        # Liveness and readiness probes
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        
        # Environment variables for carbon awareness
        env:
        - name: CARBON_AWARE_MODE
          value: "enabled"
        - name: ENERGY_EFFICIENT_MODE
          value: "true"

---
# Horizontal Pod Autoscaler with carbon consideration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: green-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: green-app
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 25
        periodSeconds: 60
            '''
        }
```

**NIST Mappings**: SA-3 (system development life cycle), CM-2 (baseline configuration)

## 5. Circular Economy Principles

### 5.1 Technology Lifecycle Management

**Standard**: Implement circular economy principles in technology management

```python
# Example: Circular economy technology lifecycle management
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class AssetStatus(Enum):
    ACTIVE = "active"
    UNDERUTILIZED = "underutilized"
    MAINTENANCE = "maintenance"
    END_OF_LIFE = "end_of_life"
    RECYCLING = "recycling"
    REFURBISHED = "refurbished"

@dataclass
class TechnologyAsset:
    id: str
    type: str  # server, laptop, mobile, etc.
    purchase_date: datetime
    last_updated: datetime
    specs: Dict
    current_utilization: float
    carbon_footprint_kg: float
    status: AssetStatus
    location: str
    lifecycle_stage: str

class CircularEconomyManager:
    """Manager for implementing circular economy principles in technology"""
    
    def __init__(self):
        self.assets = {}
        self.refurbishment_partners = []
        self.recycling_partners = []
        self.lifecycle_policies = self._initialize_lifecycle_policies()
    
    def _initialize_lifecycle_policies(self) -> Dict:
        """Initialize lifecycle management policies"""
        return {
            'utilization_threshold': 0.3,  # Below 30% utilization triggers review
            'age_thresholds': {
                'servers': {'refresh': 4, 'retire': 6},
                'laptops': {'refresh': 3, 'retire': 5},
                'mobile': {'refresh': 2, 'retire': 4},
                'networking': {'refresh': 5, 'retire': 8}
            },
            'carbon_intensity_limits': {
                'servers': 100,  # kg CO2e per year
                'laptops': 20,
                'mobile': 5
            },
            'refurbishment_criteria': {
                'min_performance_retention': 0.8,
                'max_age_years': 3,
                'min_battery_health': 0.7
            }
        }
    
    def assess_asset_lifecycle(self, asset: TechnologyAsset) -> Dict:
        """Comprehensive lifecycle assessment of technology asset"""
        age_years = (datetime.now() - asset.purchase_date).days / 365.25
        
        assessment = {
            'asset_id': asset.id,
            'current_stage': asset.lifecycle_stage,
            'age_years': age_years,
            'utilization_score': asset.current_utilization,
            'carbon_efficiency': self._calculate_carbon_efficiency(asset),
            'recommendations': [],
            'next_actions': []
        }
        
        # Age-based assessment
        thresholds = self.lifecycle_policies['age_thresholds'].get(asset.type, {})
        if age_years >= thresholds.get('retire', 10):
            assessment['recommendations'].append('Consider retirement and recycling')
            assessment['next_actions'].append('initiate_recycling_process')
        elif age_years >= thresholds.get('refresh', 5):
            assessment['recommendations'].append('Evaluate for refresh or refurbishment')
            assessment['next_actions'].append('evaluate_refurbishment')
        
        # Utilization-based assessment
        if asset.current_utilization < self.lifecycle_policies['utilization_threshold']:
            assessment['recommendations'].append('Asset is underutilized - consider reallocation')
            assessment['next_actions'].append('find_better_utilization')
        
        # Carbon efficiency assessment
        carbon_limit = self.lifecycle_policies['carbon_intensity_limits'].get(asset.type, 50)
        if asset.carbon_footprint_kg > carbon_limit:
            assessment['recommendations'].append('High carbon footprint - prioritize for replacement')
            assessment['next_actions'].append('plan_green_replacement')
        
        return assessment
    
    def _calculate_carbon_efficiency(self, asset: TechnologyAsset) -> float:
        """Calculate carbon efficiency score"""
        # Carbon per unit of useful work
        if asset.current_utilization > 0:
            return asset.carbon_footprint_kg / asset.current_utilization
        return float('inf')
    
    def plan_circular_transitions(self, assets: List[TechnologyAsset]) -> Dict:
        """Plan circular economy transitions for asset portfolio"""
        transition_plan = {
            'refurbishment_candidates': [],
            'reallocation_opportunities': [],
            'recycling_queue': [],
            'carbon_savings_potential': 0,
            'cost_savings_potential': 0
        }
        
        for asset in assets:
            assessment = self.assess_asset_lifecycle(asset)
            
            if 'evaluate_refurbishment' in assessment['next_actions']:
                refurb_potential = self._evaluate_refurbishment_potential(asset)
                if refurb_potential['viable']:
                    transition_plan['refurbishment_candidates'].append({
                        'asset': asset,
                        'potential': refurb_potential
                    })
            
            if 'find_better_utilization' in assessment['next_actions']:
                reallocation = self._find_reallocation_opportunity(asset)
                if reallocation:
                    transition_plan['reallocation_opportunities'].append(reallocation)
            
            if 'initiate_recycling_process' in assessment['next_actions']:
                recycling_plan = self._plan_recycling(asset)
                transition_plan['recycling_queue'].append(recycling_plan)
        
        # Calculate potential savings
        transition_plan['carbon_savings_potential'] = self._calculate_carbon_savings(transition_plan)
        transition_plan['cost_savings_potential'] = self._calculate_cost_savings(transition_plan)
        
        return transition_plan
    
    def _evaluate_refurbishment_potential(self, asset: TechnologyAsset) -> Dict:
        """Evaluate potential for asset refurbishment"""
        criteria = self.lifecycle_policies['refurbishment_criteria']
        age_years = (datetime.now() - asset.purchase_date).days / 365.25
        
        # Performance assessment
        performance_score = self._assess_performance_retention(asset)
        
        # Economic viability
        refurb_cost = self._estimate_refurbishment_cost(asset)
        new_cost = self._estimate_new_asset_cost(asset.type)
        cost_ratio = refurb_cost / new_cost
        
        # Environmental impact
        carbon_savings = self._calculate_refurb_carbon_savings(asset)
        
        viable = (
            performance_score >= criteria['min_performance_retention'] and
            age_years <= criteria['max_age_years'] and
            cost_ratio <= 0.6  # Refurbishment should cost less than 60% of new
        )
        
        return {
            'viable': viable,
            'performance_score': performance_score,
            'cost_ratio': cost_ratio,
            'carbon_savings_kg': carbon_savings,
            'estimated_extended_life_years': 2 if viable else 0,
            'refurbishment_actions': self._generate_refurbishment_plan(asset) if viable else []
        }
    
    def _generate_refurbishment_plan(self, asset: TechnologyAsset) -> List[str]:
        """Generate specific refurbishment actions"""
        actions = []
        
        if asset.type == 'laptop':
            actions.extend([
                'Replace battery if health < 80%',
                'Upgrade RAM to maximum supported',
                'Replace HDD with SSD if applicable',
                'Clean and reapply thermal paste',
                'Update firmware and drivers'
            ])
        elif asset.type == 'server':
            actions.extend([
                'Replace aging hard drives',
                'Add memory if cost-effective',
                'Update firmware and BIOS',
                'Replace cooling fans if necessary',
                'Verify and update security patches'
            ])
        elif asset.type == 'mobile':
            actions.extend([
                'Replace battery',
                'Factory reset and OS update',
                'Screen repair if needed',
                'Camera and sensor calibration'
            ])
        
        return actions
    
    def implement_asset_sharing_platform(self) -> Dict:
        """Implement platform for asset sharing and optimization"""
        return {
            'platform_features': {
                'asset_discovery': {
                    'description': 'Find underutilized assets across organization',
                    'implementation': '''
                    def find_underutilized_assets(utilization_threshold=0.3):
                        return [
                            asset for asset in all_assets 
                            if asset.current_utilization < utilization_threshold
                            and asset.status == AssetStatus.ACTIVE
                        ]
                    '''
                },
                'sharing_scheduler': {
                    'description': 'Schedule shared asset usage',
                    'implementation': '''
                    class AssetSharingScheduler:
                        def __init__(self):
                            self.bookings = {}
                            self.asset_availability = {}
                        
                        def book_asset(self, asset_id, user_id, start_time, duration):
                            if self.is_available(asset_id, start_time, duration):
                                booking_id = self.create_booking(
                                    asset_id, user_id, start_time, duration
                                )
                                return booking_id
                            return None
                        
                        def optimize_utilization(self):
                            # Find gaps in usage and suggest consolidation
                            for asset_id, schedule in self.bookings.items():
                                gaps = self.find_usage_gaps(schedule)
                                if gaps:
                                    self.suggest_gap_filling(asset_id, gaps)
                    '''
                },
                'carbon_tracking': {
                    'description': 'Track carbon savings from sharing',
                    'metrics': [
                        'Assets prevented from purchase',
                        'Carbon footprint reduction',
                        'Utilization improvement',
                        'Cost savings achieved'
                    ]
                }
            },
            'governance': {
                'sharing_policies': [
                    'Assets must be cleaned/sanitized between users',
                    'Minimum booking duration: 4 hours',
                    'Maximum consecutive booking: 2 weeks',
                    'Priority given to business-critical needs'
                ],
                'maintenance_responsibilities': [
                    'Shared assets get priority maintenance',
                    'Users report issues immediately',
                    'Regular health checks every 30 days'
                ]
            },
            'success_metrics': {
                'utilization_improvement': 'Target: 40% increase in asset utilization',
                'carbon_reduction': 'Target: 25% reduction in new asset purchases',
                'cost_savings': 'Target: 30% reduction in hardware costs',
                'user_satisfaction': 'Target: 85% satisfaction with shared assets'
            }
        }
    
    def create_recycling_partnership_program(self) -> Dict:
        """Create comprehensive recycling partnership program"""
        return {
            'partner_criteria': {
                'certifications_required': [
                    'R2 (Responsible Recycling)',
                    'e-Stewards certification',
                    'ISO 14001 environmental management',
                    'ISO 45001 occupational health and safety'
                ],
                'data_security_requirements': [
                    'NIST 800-88 compliant data destruction',
                    'Certificate of data destruction provided',
                    'Chain of custody documentation',
                    'Video surveillance of destruction process'
                ],
                'environmental_standards': [
                    'Zero landfill policy',
                    'Material recovery rate > 95%',
                    'Hazardous material handling certification',
                    'Carbon footprint reporting'
                ]
            },
            'process_workflow': {
                'asset_collection': [
                    'Schedule pickup with certified partner',
                    'Generate asset inventory with serial numbers',
                    'Apply tamper-evident seals',
                    'Document chain of custody'
                ],
                'data_destruction': [
                    'Remove all storage devices',
                    'Perform DoD 5220.22-M wipe (minimum)',
                    'Physical destruction for sensitive devices',
                    'Obtain destruction certificates'
                ],
                'material_recovery': [
                    'Disassemble devices into material streams',
                    'Separate precious metals, rare earth elements',
                    'Process plastics for recycling',
                    'Recover usable components for refurbishment'
                ],
                'reporting': [
                    'Monthly recycling reports',
                    'Carbon impact calculations',
                    'Material recovery percentages',
                    'Cost savings documentation'
                ]
            },
            'impact_tracking': {
                'environmental_metrics': [
                    'Total weight diverted from landfill',
                    'Precious metals recovered',
                    'CO2 emissions avoided',
                    'Energy savings from recycling vs. mining'
                ],
                'business_metrics': [
                    'Revenue from material recovery',
                    'Cost avoidance from proper disposal',
                    'Compliance risk mitigation',
                    'Brand reputation enhancement'
                ]
            }
        }
```

**NIST Mappings**: MP-6 (media sanitization), SA-12 (supply chain protection)

## 6. Green DevOps Practices

### 6.1 Sustainable CI/CD Pipelines

**Standard**: Implement energy-efficient CI/CD practices

```python
# Example: Green CI/CD pipeline optimization
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import yaml

class GreenCICDOptimizer:
    """Optimizer for sustainable CI/CD practices"""
    
    def __init__(self):
        self.carbon_intensity_api = "https://api.carbonintensity.org.uk"
        self.pipeline_cache = {}
        self.energy_metrics = {}
        
    def optimize_pipeline_scheduling(self, pipelines: List[Dict]) -> Dict:
        """Optimize CI/CD pipeline scheduling for carbon efficiency"""
        optimization_plan = {
            'immediate_execution': [],
            'scheduled_execution': [],
            'carbon_savings_estimated': 0,
            'optimization_strategies': []
        }
        
        # Categorize pipelines by urgency and carbon sensitivity
        critical_pipelines = [p for p in pipelines if p.get('priority') == 'critical']
        batch_pipelines = [p for p in pipelines if p.get('type') == 'batch']
        test_pipelines = [p for p in pipelines if p.get('type') == 'test']
        
        # Execute critical pipelines immediately
        for pipeline in critical_pipelines:
            optimization_plan['immediate_execution'].append({
                'pipeline_id': pipeline['id'],
                'reason': 'Critical priority - immediate execution required',
                'estimated_carbon_kg': self._estimate_pipeline_carbon(pipeline)
            })
        
        # Schedule non-critical pipelines for low-carbon windows
        for pipeline in batch_pipelines + test_pipelines:
            optimal_window = self._find_low_carbon_window(pipeline)
            optimization_plan['scheduled_execution'].append({
                'pipeline_id': pipeline['id'],
                'current_carbon_intensity': optimal_window['current_intensity'],
                'scheduled_intensity': optimal_window['scheduled_intensity'],
                'execution_time': optimal_window['execution_time'],
                'carbon_savings_kg': optimal_window['carbon_savings']
            })
        
        # Generate optimization strategies
        optimization_plan['optimization_strategies'] = self._generate_pipeline_optimizations()
        
        return optimization_plan
    
    def _estimate_pipeline_carbon(self, pipeline: Dict) -> float:
        """Estimate carbon footprint of pipeline execution"""
        # Base estimates for different pipeline types
        carbon_factors = {
            'build': 0.05,      # kg CO2 per minute
            'test': 0.03,       # kg CO2 per minute
            'deploy': 0.02,     # kg CO2 per minute
            'security_scan': 0.04,  # kg CO2 per minute
        }
        
        total_carbon = 0
        for stage in pipeline.get('stages', []):
            stage_type = stage.get('type', 'build')
            duration_minutes = stage.get('estimated_duration', 5)
            parallel_runners = stage.get('parallel_runners', 1)
            
            stage_carbon = (
                carbon_factors.get(stage_type, 0.03) * 
                duration_minutes * 
                parallel_runners
            )
            total_carbon += stage_carbon
        
        return total_carbon
    
    def _find_low_carbon_window(self, pipeline: Dict) -> Dict:
        """Find optimal execution window with low carbon intensity"""
        # Current carbon intensity (example values)
        current_intensity = 450  # g CO2/kWh
        
        # Find next low-carbon window (typically 2-8 hours ahead)
        low_carbon_times = [
            {'time': datetime.now() + timedelta(hours=2), 'intensity': 320},
            {'time': datetime.now() + timedelta(hours=4), 'intensity': 280},
            {'time': datetime.now() + timedelta(hours=6), 'intensity': 350},
        ]
        
        best_window = min(low_carbon_times, key=lambda x: x['intensity'])
        pipeline_carbon = self._estimate_pipeline_carbon(pipeline)
        
        # Calculate savings
        current_emissions = pipeline_carbon * (current_intensity / 1000)
        scheduled_emissions = pipeline_carbon * (best_window['intensity'] / 1000)
        carbon_savings = current_emissions - scheduled_emissions
        
        return {
            'current_intensity': current_intensity,
            'scheduled_intensity': best_window['intensity'],
            'execution_time': best_window['time'],
            'carbon_savings': carbon_savings
        }
    
    def _generate_pipeline_optimizations(self) -> List[Dict]:
        """Generate specific optimization recommendations"""
        return [
            {
                'optimization': 'Intelligent Build Caching',
                'description': 'Cache build artifacts and dependencies to reduce computation',
                'implementation': '''
                # Example: Dockerfile with multi-stage builds and caching
                FROM node:16-alpine AS dependencies
                WORKDIR /app
                COPY package*.json ./
                RUN npm ci --only=production && npm cache clean --force
                
                FROM node:16-alpine AS build
                WORKDIR /app
                COPY package*.json ./
                RUN npm ci
                COPY . .
                RUN npm run build
                
                FROM nginx:alpine
                COPY --from=build /app/build /usr/share/nginx/html
                ''',
                'carbon_reduction': '30-60% reduction in build time and energy'
            },
            {
                'optimization': 'Parallel Test Execution',
                'description': 'Run tests in parallel to reduce total execution time',
                'implementation': '''
                # Example: Parallel test configuration
                test:
                  parallel: 4
                  script:
                    - npm run test:unit -- --coverage
                    - npm run test:integration
                    - npm run test:e2e
                  coverage: '/Coverage: \d+\.\d+%/'
                ''',
                'carbon_reduction': '40-70% reduction in test execution time'
            },
            {
                'optimization': 'Smart Deployment Strategies',
                'description': 'Use efficient deployment patterns to minimize resource usage',
                'implementation': '''
                # Example: Rolling deployment with resource limits
                deployment:
                  strategy:
                    type: rolling
                    rolling_update:
                      max_surge: 1
                      max_unavailable: 0
                  resource_limits:
                    cpu: 500m
                    memory: 512Mi
                  auto_scaling:
                    min_replicas: 1
                    max_replicas: 5
                    target_cpu: 70%
                ''',
                'carbon_reduction': '20-40% reduction in deployment resource usage'
            }
        ]
    
    def implement_green_pipeline_config(self) -> str:
        """Generate green CI/CD pipeline configuration"""
        config = {
            'variables': {
                'CARBON_AWARE_MODE': 'enabled',
                'ENERGY_EFFICIENT_BUILD': 'true',
                'CACHE_OPTIMIZATION': 'aggressive'
            },
            'stages': ['validate', 'build', 'test', 'security', 'deploy'],
            'before_script': [
                'export DOCKER_BUILDKIT=1',
                'export COMPOSE_DOCKER_CLI_BUILD=1'
            ],
            'carbon_validation': {
                'stage': 'validate',
                'script': [
                    'python scripts/carbon_check.py',
                    'if [ $CARBON_INTENSITY -gt 400 ]; then echo "High carbon intensity detected"; fi'
                ],
                'rules': [
                    {'if': '$CI_PIPELINE_SOURCE == "schedule"'},
                    {'if': '$CARBON_AWARE_MODE == "enabled"'}
                ]
            },
            'efficient_build': {
                'stage': 'build',
                'script': [
                    'docker build --cache-from $CI_REGISTRY_IMAGE:latest --tag $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .',
                    'docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA'
                ],
                'cache': {
                    'key': '$CI_COMMIT_REF_SLUG',
                    'paths': [
                        'node_modules/',
                        '.npm/',
                        'target/',
                        '.gradle/'
                    ]
                },
                'only': {
                    'changes': [
                        'src/**/*',
                        'package*.json',
                        'Dockerfile'
                    ]
                }
            },
            'parallel_testing': {
                'stage': 'test',
                'parallel': 3,
                'script': [
                    'npm run test:parallel -- --shard=$CI_NODE_INDEX/$CI_NODE_TOTAL'
                ],
                'artifacts': {
                    'reports': {
                        'junit': 'test-results.xml',
                        'coverage_report': {
                            'coverage_format': 'cobertura',
                            'path': 'coverage/cobertura-coverage.xml'
                        }
                    }
                }
            },
            'carbon_aware_deploy': {
                'stage': 'deploy',
                'script': [
                    'python scripts/check_carbon_window.py',
                    'if [ $CARBON_OK == "true" ]; then kubectl apply -f k8s/; fi'
                ],
                'environment': {
                    'name': 'production',
                    'url': 'https://myapp.com'
                },
                'rules': [
                    {'if': '$CI_COMMIT_BRANCH == "main"'},
                    {'when': 'manual', 'allow_failure': False}
                ]
            }
        }
        
        return yaml.dump(config, default_flow_style=False)
    
    def create_carbon_monitoring_dashboard(self) -> Dict:
        """Create dashboard for monitoring CI/CD carbon impact"""
        return {
            'metrics': {
                'real_time': [
                    'Current carbon intensity',
                    'Active pipeline carbon footprint',
                    'Queue carbon footprint',
                    'Regional carbon intensity comparison'
                ],
                'historical': [
                    'Daily carbon emissions trend',
                    'Pipeline efficiency improvements',
                    'Carbon savings from optimizations',
                    'Peak vs off-peak usage patterns'
                ],
                'predictive': [
                    'Upcoming carbon intensity forecast',
                    'Optimal execution windows',
                    'Projected monthly carbon budget',
                    'Seasonal carbon trends'
                ]
            },
            'alerts': {
                'high_carbon_intensity': {
                    'threshold': '> 400 g CO2/kWh',
                    'action': 'Defer non-critical pipelines',
                    'notification': 'Slack + Email'
                },
                'carbon_budget_warning': {
                    'threshold': '> 80% of monthly budget',
                    'action': 'Require approval for new pipelines',
                    'notification': 'Management dashboard'
                },
                'inefficient_pipeline': {
                    'threshold': '> 2x average carbon per build',
                    'action': 'Trigger optimization review',
                    'notification': 'Development team'
                }
            },
            'automation': [
                'Auto-defer pipelines during high carbon periods',
                'Schedule batch operations for green windows',
                'Auto-optimize pipeline configurations',
                'Generate weekly carbon reports'
            ]
        }

# Green infrastructure monitoring
class GreenInfrastructureMonitor:
    """Monitor infrastructure for sustainability metrics"""
    
    def __init__(self):
        self.sustainability_kpis = {
            'energy_efficiency': 'PUE (Power Usage Effectiveness)',
            'carbon_intensity': 'kg CO2e per service request',
            'resource_utilization': 'CPU/Memory/Storage efficiency',
            'renewable_energy': 'Percentage renewable energy usage',
            'waste_reduction': 'Hardware lifecycle optimization'
        }
    
    def calculate_sustainability_score(self, infrastructure_metrics: Dict) -> Dict:
        """Calculate comprehensive sustainability score"""
        scores = {}
        
        # Energy Efficiency Score (0-100)
        pue = infrastructure_metrics.get('pue', 2.0)
        energy_score = max(0, min(100, (2.5 - pue) * 100))
        scores['energy_efficiency'] = energy_score
        
        # Carbon Intensity Score (0-100)
        carbon_intensity = infrastructure_metrics.get('carbon_intensity_g_kwh', 500)
        carbon_score = max(0, min(100, (600 - carbon_intensity) / 6))
        scores['carbon_intensity'] = carbon_score
        
        # Resource Utilization Score (0-100)
        cpu_util = infrastructure_metrics.get('avg_cpu_utilization', 0.3)
        memory_util = infrastructure_metrics.get('avg_memory_utilization', 0.4)
        storage_util = infrastructure_metrics.get('avg_storage_utilization', 0.5)
        
        # Optimal utilization is around 70-80%
        cpu_score = max(0, 100 - abs(cpu_util - 0.75) * 200)
        memory_score = max(0, 100 - abs(memory_util - 0.75) * 200)
        storage_score = max(0, 100 - abs(storage_util - 0.75) * 200)
        
        scores['resource_utilization'] = (cpu_score + memory_score + storage_score) / 3
        
        # Renewable Energy Score (0-100)
        renewable_percentage = infrastructure_metrics.get('renewable_energy_percentage', 0)
        scores['renewable_energy'] = renewable_percentage
        
        # Hardware Lifecycle Score (0-100)
        avg_asset_age = infrastructure_metrics.get('avg_hardware_age_years', 5)
        asset_utilization = infrastructure_metrics.get('avg_asset_utilization', 0.5)
        lifecycle_score = min(100, asset_utilization * 100) * max(0, (6 - avg_asset_age) / 6)
        scores['hardware_lifecycle'] = lifecycle_score
        
        # Calculate overall score
        weights = {
            'energy_efficiency': 0.25,
            'carbon_intensity': 0.25,
            'resource_utilization': 0.20,
            'renewable_energy': 0.20,
            'hardware_lifecycle': 0.10
        }
        
        overall_score = sum(scores[metric] * weight for metric, weight in weights.items())
        
        return {
            'overall_score': overall_score,
            'individual_scores': scores,
            'grade': self._get_sustainability_grade(overall_score),
            'recommendations': self._generate_sustainability_recommendations(scores)
        }
    
    def _get_sustainability_grade(self, score: float) -> str:
        """Convert sustainability score to letter grade"""
        if score >= 90:
            return 'A+'
        elif score >= 80:
            return 'A'
        elif score >= 70:
            return 'B'
        elif score >= 60:
            return 'C'
        elif score >= 50:
            return 'D'
        else:
            return 'F'
    
    def _generate_sustainability_recommendations(self, scores: Dict) -> List[str]:
        """Generate specific recommendations based on scores"""
        recommendations = []
        
        if scores['energy_efficiency'] < 70:
            recommendations.append(
                "Improve energy efficiency: optimize cooling, use energy-efficient hardware"
            )
        
        if scores['carbon_intensity'] < 60:
            recommendations.append(
                "Reduce carbon intensity: migrate to regions with cleaner energy"
            )
        
        if scores['resource_utilization'] < 60:
            recommendations.append(
                "Optimize resource utilization: implement auto-scaling, rightsize instances"
            )
        
        if scores['renewable_energy'] < 50:
            recommendations.append(
                "Increase renewable energy usage: choose green cloud providers/regions"
            )
        
        if scores['hardware_lifecycle'] < 70:
            recommendations.append(
                "Optimize hardware lifecycle: implement asset sharing, extend useful life"
            )
        
        return recommendations
```

**NIST Mappings**: SA-3 (system development life cycle), CM-7 (least functionality)

## 7. Best Practices Summary

### 7.1 Implementation Guidelines

1. **Carbon Measurement**: Implement comprehensive carbon footprint tracking
2. **Energy Optimization**: Optimize algorithms and resource usage for efficiency
3. **Green Infrastructure**: Choose low-carbon cloud regions and efficient hardware
4. **Circular Economy**: Extend hardware lifecycle and implement sharing programs
5. **Sustainable DevOps**: Optimize CI/CD pipelines for energy efficiency

### 7.2 Key Performance Indicators

- Carbon intensity per user transaction
- Energy efficiency improvements over time
- Hardware utilization rates and lifecycle extension
- Renewable energy percentage
- Waste reduction metrics

### 7.3 Compliance and Reporting

1. **Environmental Standards**: ISO 14001, EMAS
2. **Carbon Reporting**: GHG Protocol, CDP disclosure
3. **Circular Economy**: Ellen MacArthur Foundation principles
4. **Energy Management**: ISO 50001

## 8. Conclusion

These sustainability and green computing standards provide a comprehensive framework for reducing the environmental impact of technology operations while maintaining performance and reliability. Regular assessment and continuous improvement of these practices ensure ongoing environmental responsibility and operational excellence.

The implementation of these standards supports both environmental stewardship and business objectives through cost savings, risk reduction, and enhanced reputation. Organizations should regularly review and update these practices to align with evolving environmental regulations and technological advances.