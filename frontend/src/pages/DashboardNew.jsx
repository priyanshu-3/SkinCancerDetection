import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import Sidebar from '../components/Sidebar'
import { 
  Activity, 
  TrendingUp, 
  AlertTriangle, 
  CheckCircle,
  Clock,
  Upload,
  Camera,
  Eye,
  Download,
  BarChart3
} from 'lucide-react'

export default function DashboardNew() {
  const navigate = useNavigate()
  const [collapsed, setCollapsed] = useState(false)
  const [stats, setStats] = useState(null)
  const [recentAnalyses, setRecentAnalyses] = useState([])
  const [loading, setLoading] = useState(true)

  // Fetch dashboard data
  useEffect(() => {
    fetchDashboardData()
  }, [])

  const fetchDashboardData = async () => {
    try {
      // Fetch stats
      const statsRes = await fetch('http://localhost:5001/api/history/stats', {
        credentials: 'include'
      })
      if (statsRes.ok) {
        const contentType = statsRes.headers.get('content-type')
        if (contentType && contentType.includes('application/json')) {
          const data = await statsRes.json()
          setStats(data.stats)
        } else {
          console.log('Stats response is not JSON, likely redirected to login')
        }
      }

      // Fetch recent analyses
      const historyRes = await fetch('http://localhost:5001/api/history', {
        credentials: 'include'
      })
      if (historyRes.ok) {
        const contentType = historyRes.headers.get('content-type')
        if (contentType && contentType.includes('application/json')) {
          const data = await historyRes.json()
          setRecentAnalyses(data.history.slice(0, 5))
        } else {
          console.log('History response is not JSON, likely redirected to login')
        }
      }
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error)
    } finally {
      setLoading(false)
    }
  }

  const getRiskColor = (diagnosis) => {
    const highRisk = ['melanoma', 'basal cell carcinoma']
    const mediumRisk = ['actinic keratoses']
    
    if (highRisk.some(d => diagnosis.toLowerCase().includes(d.toLowerCase()))) {
      return 'text-red-600 bg-red-100'
    } else if (mediumRisk.some(d => diagnosis.toLowerCase().includes(d.toLowerCase()))) {
      return 'text-orange-600 bg-orange-100'
    }
    return 'text-green-600 bg-green-100'
  }

  const StatCard = ({ icon: Icon, label, value, trend, color }) => (
    <div className="bg-white rounded-xl shadow-md p-6 border-l-4" style={{ borderColor: color }}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-gray-500 text-sm font-medium">{label}</p>
          <h3 className="text-3xl font-bold mt-2" style={{ color }}>{value}</h3>
          {trend && (
            <p className="text-sm text-gray-600 mt-2 flex items-center gap-1">
              <TrendingUp className="w-4 h-4" />
              {trend}
            </p>
          )}
        </div>
        <div className="w-14 h-14 rounded-full flex items-center justify-center" style={{ backgroundColor: color + '20' }}>
          <Icon className="w-7 h-7" style={{ color }} />
        </div>
      </div>
    </div>
  )

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <Sidebar collapsed={collapsed} setCollapsed={setCollapsed} />

      {/* Main Content */}
      <div className={`flex-1 transition-all duration-300 ${collapsed ? 'ml-20' : 'ml-64'}`}>
        {/* Header */}
        <header className="bg-white shadow-sm border-b border-gray-200">
          <div className="px-8 py-4">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Dashboard Overview</h1>
                <p className="text-sm text-gray-600 mt-1">Welcome back! Here's your summary</p>
              </div>
              <button
                onClick={() => navigate('/analysis')}
                className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors shadow-md"
              >
                <Activity className="w-5 h-5" />
                New Analysis
              </button>
            </div>
          </div>
        </header>

        {/* Dashboard Content */}
        <div className="p-8 overflow-y-auto h-[calc(100vh-88px)]">
          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
            <StatCard
              icon={Activity}
              label="Total Analyses"
              value={stats?.total_analyses || 0}
              color="#3B82F6"
            />
            <StatCard
              icon={CheckCircle}
              label="Avg Confidence"
              value={stats ? `${stats.avg_confidence.toFixed(1)}%` : '0%'}
              color="#10B981"
            />
            <StatCard
              icon={Clock}
              label="This Week"
              value={recentAnalyses.length}
              color="#F59E0B"
            />
          </div>

          {/* Main Grid Layout */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left Column - Charts and Analytics */}
            <div className="lg:col-span-2 space-y-6">
              {/* Diagnosis Distribution */}
              <div className="bg-white rounded-xl shadow-md p-6">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-xl font-bold text-gray-900 flex items-center gap-2">
                    <BarChart3 className="w-6 h-6 text-blue-600" />
                    Diagnosis Distribution
                  </h2>
                </div>
                {stats?.diagnosis_breakdown ? (
                  <div className="space-y-4">
                    {Object.entries(stats.diagnosis_breakdown).map(([diagnosis, count]) => (
                      <div key={diagnosis} className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="font-medium text-gray-700">{diagnosis}</span>
                          <span className="text-gray-600">{count} cases</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-3">
                          <div
                            className="bg-gradient-to-r from-blue-500 to-blue-600 h-3 rounded-full transition-all"
                            style={{ width: `${(count / stats.total_analyses) * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    No analysis data available yet
                  </div>
                )}
              </div>

              {/* Recent Activity Table */}
              <div className="bg-white rounded-xl shadow-md p-6">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-xl font-bold text-gray-900 flex items-center gap-2">
                    <Clock className="w-6 h-6 text-blue-600" />
                    Recent Analyses
                  </h2>
                  <button
                    onClick={() => navigate('/history')}
                    className="text-blue-600 hover:text-blue-700 text-sm font-medium"
                  >
                    View All →
                  </button>
                </div>
                {recentAnalyses.length > 0 ? (
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b border-gray-200">
                          <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700">Patient</th>
                          <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700">Diagnosis</th>
                          <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700">Confidence</th>
                          <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700">Date</th>
                          <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700">Action</th>
                        </tr>
                      </thead>
                      <tbody>
                        {recentAnalyses.map((analysis) => (
                          <tr key={analysis.id} className="border-b border-gray-100 hover:bg-gray-50">
                            <td className="py-3 px-4">
                              <div>
                                <div className="font-medium text-gray-900">{analysis.patient_name}</div>
                                <div className="text-sm text-gray-500">{analysis.age} yrs, {analysis.gender}</div>
                              </div>
                            </td>
                            <td className="py-3 px-4">
                              <span className={`px-3 py-1 rounded-full text-xs font-semibold ${getRiskColor(analysis.diagnosis)}`}>
                                {analysis.diagnosis}
                              </span>
                            </td>
                            <td className="py-3 px-4 text-sm text-gray-900">
                              {analysis.confidence.toFixed(1)}%
                            </td>
                            <td className="py-3 px-4 text-sm text-gray-600">
                              {analysis.created_at ? new Date(analysis.created_at).toLocaleDateString() : 'N/A'}
                            </td>
                            <td className="py-3 px-4">
                              <button
                                onClick={() => navigate(`/history?view=${analysis.id}`)}
                                className="text-blue-600 hover:text-blue-700"
                              >
                                <Eye className="w-5 h-5" />
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    No recent analyses found
                  </div>
                )}
              </div>
            </div>

            {/* Right Column - Quick Actions */}
            <div className="space-y-6">
              {/* Quick Actions Card */}
              <div className="bg-white rounded-xl shadow-md p-6">
                <h2 className="text-xl font-bold text-gray-900 mb-4">Quick Actions</h2>
                <div className="space-y-3">
                  <button
                    onClick={() => navigate('/analysis')}
                    className="w-full flex items-center gap-3 px-4 py-3 bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-lg hover:from-blue-700 hover:to-blue-800 transition-all shadow-md"
                  >
                    <Upload className="w-5 h-5" />
                    <span className="font-medium">Upload Image</span>
                  </button>
                  <button
                    onClick={() => navigate('/analysis?mode=camera')}
                    className="w-full flex items-center gap-3 px-4 py-3 bg-gradient-to-r from-green-600 to-green-700 text-white rounded-lg hover:from-green-700 hover:to-green-800 transition-all shadow-md"
                  >
                    <Camera className="w-5 h-5" />
                    <span className="font-medium">Use Camera</span>
                  </button>
                  <button
                    onClick={() => navigate('/history')}
                    className="w-full flex items-center gap-3 px-4 py-3 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-all"
                  >
                    <Clock className="w-5 h-5" />
                    <span className="font-medium">View History</span>
                  </button>
                </div>
              </div>

              {/* Risk Alert Card */}
              <div className="bg-gradient-to-br from-red-50 to-red-100 rounded-xl shadow-md p-6 border border-red-200">
                <div className="flex items-start gap-3">
                  <AlertTriangle className="w-6 h-6 text-red-600 flex-shrink-0 mt-1" />
                  <div>
                    <h3 className="font-bold text-red-900 mb-2">High Risk Cases</h3>
                    <p className="text-sm text-red-800 mb-3">
                      {stats?.diagnosis_breakdown?.melanoma || 0} melanoma cases detected
                    </p>
                    <button
                      onClick={() => navigate('/history?filter=high-risk')}
                      className="text-sm text-red-600 hover:text-red-700 font-medium"
                    >
                      Review Cases →
                    </button>
                  </div>
                </div>
              </div>

              {/* System Status Card */}
              <div className="bg-white rounded-xl shadow-md p-6">
                <h3 className="font-bold text-gray-900 mb-4">System Status</h3>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">AI Model</span>
                    <span className="flex items-center gap-2 text-sm font-medium text-green-600">
                      <div className="w-2 h-2 bg-green-600 rounded-full"></div>
                      Active
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Database</span>
                    <span className="flex items-center gap-2 text-sm font-medium text-green-600">
                      <div className="w-2 h-2 bg-green-600 rounded-full"></div>
                      Connected
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">API Status</span>
                    <span className="flex items-center gap-2 text-sm font-medium text-green-600">
                      <div className="w-2 h-2 bg-green-600 rounded-full"></div>
                      Operational
                    </span>
                  </div>
                </div>
              </div>

              {/* Info Card */}
              <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl shadow-md p-6 border border-blue-200">
                <h3 className="font-bold text-blue-900 mb-2">💡 Quick Tip</h3>
                <p className="text-sm text-blue-800">
                  For best results, ensure images are well-lit and focused on the lesion area. High-quality images lead to more accurate predictions.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

