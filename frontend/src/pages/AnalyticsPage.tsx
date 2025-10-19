import { useEffect, useMemo, useState } from 'react';
import { analytics, getAllProducts, normalizeCategories, parsePrice } from '@/lib/api';
import type { Analytics, Product } from '@/types';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
  Legend,
  Title,
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend, Title);

export default function AnalyticsPage() {
  const [summary, setSummary] = useState<Analytics | null>(null);
  const [products, setProducts] = useState<Product[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    const fetchAnalytics = async () => {
      setLoading(true);
      setErr(null);
      try {
        const [analyticsData, productData] = await Promise.all([
          analytics(),
          getAllProducts(),
        ]);

        setSummary(analyticsData);
        setProducts(productData.products || []);
      } catch (e: any) {
        console.error(e);
        setErr(e.message || 'Failed to load analytics');
      } finally {
        setLoading(false);
      }
    };
    fetchAnalytics();
  }, []);

  // Compute top brands
  const brandTop = useMemo(() => {
    const map = new Map<string, number>();
    for (const p of products) {
      if (p.brand) map.set(p.brand, (map.get(p.brand) || 0) + 1);
    }
    return Array.from(map.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10);
  }, [products]);

  // Compute top categories
  const categoryTop = useMemo(() => {
    const map = new Map<string, number>();
    for (const p of products) {
      const cats = normalizeCategories(p.categories);
      for (const c of cats) {
        map.set(c, (map.get(c) || 0) + 1);
      }
    }
    return Array.from(map.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10);
  }, [products]);

  if (loading) return <div className="page">Loading analytics...</div>;
  if (err) return <div className="page error">Error: {err}</div>;

  return (
    <div className="page analytics-page">
      <h2>Dataset Analytics</h2>

      {/* KPIs Section */}
      {summary && (
        <div className="kpis">
          <div className="kpi">
            <div className="kpi-label">Total Products</div>
            <div className="kpi-value">
              {summary.total_products ?? products.length}
            </div>
          </div>

          <div className="kpi">
            <div className="kpi-label">Avg. Price</div>
            <div className="kpi-value">
              {summary.price_statistics?.mean
                ? `$${summary.price_statistics.mean.toFixed(2)}`
                : '—'}
            </div>
          </div>

          <div className="kpi">
            <div className="kpi-label">Brands</div>
            <div className="kpi-value">
              {Object.keys(summary.brand_distribution || {}).length}
            </div>
          </div>

          <div className="kpi">
            <div className="kpi-label">Categories</div>
            <div className="kpi-value">
              {Object.keys(summary.categories_distribution || {}).length}
            </div>
          </div>
        </div>
      )}

      {/* Charts Section */}
      <div className="charts">
        <div className="chart-card">
          <h3>Top Categories</h3>
          <Bar
            data={{
              labels: categoryTop.map(([k]) => k),
              datasets: [
                {
                  label: 'Count',
                  data: categoryTop.map(([, v]) => v),
                  backgroundColor: '#4F46E5',
                },
              ],
            }}
            options={{
              responsive: true,
              plugins: {
                legend: { display: false },
                title: { display: true, text: 'Category Distribution' },
              },
              scales: { x: { ticks: { maxRotation: 60 } } },
            }}
          />
        </div>

        <div className="chart-card">
          <h3>Top Brands</h3>
          <Bar
            data={{
              labels: brandTop.map(([k]) => k),
              datasets: [
                {
                  label: 'Count',
                  data: brandTop.map(([, v]) => v),
                  backgroundColor: '#10B981',
                },
              ],
            }}
            options={{
              responsive: true,
              plugins: {
                legend: { display: false },
                title: { display: true, text: 'Brand Distribution' },
              },
              scales: { x: { ticks: { maxRotation: 60 } } },
            }}
          />
        </div>
      </div>

      {/* Product Table */}
      <div className="table">
        <h3>Sample Products</h3>
        <div className="table-head">
          <div>Title</div>
          <div>Brand</div>
          <div>Price</div>
          <div>Material</div>
          <div>Color</div>
        </div>
        {products.slice(0, 25).map((p) => (
          <div className="table-row" key={p.uniq_id}>
            <div title={p.title}>{p.title}</div>
            <div>{p.brand ?? '—'}</div>
            <div>
              {(() => {
                const priceNum = parsePrice(p.price);
                return priceNum !== null ? `$${priceNum.toFixed(2)}` : '—';
              })()}
            </div>
            <div>{p.material ?? '—'}</div>
            <div>{p.color ?? '—'}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
