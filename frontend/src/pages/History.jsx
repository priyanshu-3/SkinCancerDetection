import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import Sidebar from '../components/Sidebar'
import { 
  Search, 
  Download, 
  Filter, 
  Calendar, 
  Eye, 
  AlertCircle,
  FileText,
  ChevronDown,
  ChevronUp,
  Loader,
  History as HistoryIcon
} from 'lucide-react'

export default function History() {
  const navigate = useNavigate()
  const [collapsed, setCollapsed] = useState(false)
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [selectedDiagnosis, setSelectedDiagnosis] = useState('')
  const [showFilters, setShowFilters] = useState(false)
  const [selectedRecord, setSelectedRecord] = useState(null)
  const [showDetailModal, setShowDetailModal] = useState(false)
  const [stats, setStats] = useState(null)
  const [isAuthenticated, setIsAuthenticated] = useState(false)

  // Check authentication status
  const checkAuth = async () => {
    try {
      const response = await fetch('http://localhost:5001/api/history/stats', {
        credentials: 'include'
      })
      
      if (response.ok) {
        setIsAuthenticated(true)
        return true
      } else if (response.status === 401 || response.status === 403) {
        setIsAuthenticated(false)
        return false
      } else {
        // Check if response is HTML (redirect to login)
        const contentType = response.headers.get('content-type')
        if (contentType && contentType.includes('text/html')) {
          setIsAuthenticated(false)
          return false
        }
        setIsAuthenticated(true)
        return true
      }
    } catch (err) {
      setIsAuthenticated(false)
      return false
    }
  }

  // Fetch history data
  const fetchHistory = async () => {
    setLoading(true)
    setError(null)
    
    try {
      // Build query string
      const params = new URLSearchParams()
      if (searchTerm) params.append('search', searchTerm)
      if (startDate) params.append('start_date', startDate)
      if (endDate) params.append('end_date', endDate)
      if (selectedDiagnosis) params.append('diagnosis', selectedDiagnosis)
      
      const response = await fetch(`http://localhost:5001/api/history?${params}`, {
        credentials: 'include'
      })
      
      if (!response.ok) {
        if (response.status === 401 || response.status === 403) {
          throw new Error('Authentication required. Please log in.')
        }
        throw new Error('Failed to fetch history')
      }
      
      const data = await response.json()
      setHistory(data.history || [])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // Fetch statistics
  const fetchStats = async () => {
    try {
      const response = await fetch('http://localhost:5001/api/history/stats', {
        credentials: 'include'
      })
      
      if (response.ok) {
        const data = await response.json()
        setStats(data.stats)
      } else if (response.status === 401 || response.status === 403) {
        console.log('Authentication required for stats')
        setIsAuthenticated(false)
      } else {
        console.error('Failed to fetch stats:', response.status)
      }
    } catch (err) {
      console.error('Failed to fetch stats:', err)
      setIsAuthenticated(false)
    }
  }

  // Load data on mount
  useEffect(() => {
    const loadData = async () => {
      const authStatus = await checkAuth()
      if (authStatus) {
        fetchHistory()
        fetchStats()
      } else {
        setError('Authentication required. Please log in to view history.')
        setLoading(false)
      }
    }
    loadData()
  }, [])

  // Handle search/filter
  const handleApplyFilters = () => {
    fetchHistory()
  }

  // Handle reset filters
  const handleResetFilters = () => {
    setSearchTerm('')
    setStartDate('')
    setEndDate('')
    setSelectedDiagnosis('')
    setTimeout(() => fetchHistory(), 100)
  }

  const formatMarkdown = (text) => {
    if (!text) return ''
    
    return text
      // Remove markdown headers (# ## ###)
      .replace(/^#{1,3}\s*/gm, '')
      // Remove bullet point markers (- and *)
      .replace(/^[\-\*]\s*/gm, '')
      // Convert *text** pattern (single asterisk start, double asterisk end) to bold
      .replace(/\*([^*]+)\*\*/g, '<strong>$1</strong>')
      // Convert **text** to <strong>text</strong>
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      // Convert remaining single * to bold (if not already handled)
      .replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, '<strong>$1</strong>')
      // Remove single asterisk at start of line (like *Follow‑Up:)
      .replace(/^\*\s*/gm, '')
      // Remove warning emoji
      .replace(/⚠️\s*/g, '')
      // Handle line breaks
      .replace(/\n\n/g, '<br/><br/>')
      .replace(/\n/g, '<br/>')
  }

  // Handle CSV export
  const handleExportCSV = async () => {
    try {
      const response = await fetch('http://localhost:5001/api/history/export', {
        credentials: 'include'
      })
      
      if (!response.ok) {
        if (response.status === 401 || response.status === 403) {
          throw new Error('Authentication required. Please log in.')
        }
        throw new Error('Failed to export CSV')
      }
      
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `patient_history_${new Date().toISOString().split('T')[0]}.csv`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (err) {
      alert('Failed to export CSV: ' + err.message)
    }
  }

  // Handle view details
  const handleViewDetails = async (recordId) => {
    try {
      const response = await fetch(`http://localhost:5001/api/history/${recordId}`, {
        credentials: 'include'
      })
      
      if (!response.ok) {
        if (response.status === 401 || response.status === 403) {
          throw new Error('Authentication required. Please log in.')
        }
        throw new Error('Failed to fetch details')
      }
      
      const data = await response.json()
      setSelectedRecord(data.analysis)
      setShowDetailModal(true)
    } catch (err) {
      alert('Failed to load details: ' + err.message)
    }
  }

  // Get risk level color
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

  // Get unique diagnoses for filter
  const uniqueDiagnoses = [...new Set(history.map(h => h.diagnosis))]

  // Show login prompt if not authenticated
  if (!isAuthenticated && !loading) {
    return (
      <div className="flex h-screen bg-gray-50">
        {/* Sidebar */}
        <Sidebar collapsed={collapsed} setCollapsed={setCollapsed} />

        {/* Main Content */}
        <div className={`flex-1 transition-all duration-300 ${collapsed ? 'ml-20' : 'ml-64'}`}>
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <AlertCircle className="h-16 w-16 text-red-500 mx-auto mb-4" />
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Authentication Required</h2>
              <p className="text-gray-600 mb-6">Please log in to view patient history.</p>
              <button
                onClick={() => navigate('/login')}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Go to Login
              </button>
            </div>
          </div>
        </div>
      </div>
    )
  }

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
                <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
                  <HistoryIcon className="w-7 h-7 text-blue-600" />
                  Patient History
                </h1>
                <p className="text-sm text-gray-600 mt-1">
                  View and manage all patient analysis records
                </p>
              </div>
              <button
                onClick={handleExportCSV}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors shadow-md"
              >
                <Download className="w-4 h-4" />
                Export CSV
              </button>
            </div>

            {/* Statistics Cards */}
            {stats && (
              <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-blue-50 rounded-lg p-4 border-l-4 border-blue-500">
                  <div className="text-sm text-blue-600 font-medium">Total Analyses</div>
                  <div className="text-2xl font-bold text-blue-900 mt-1">{stats.total_analyses}</div>
                </div>
              <div className="bg-green-50 rounded-lg p-4 border-l-4 border-green-500">
                <div className="text-sm text-green-600 font-medium">Avg Confidence</div>
                <div className="text-2xl font-bold text-green-900 mt-1">{stats.avg_confidence.toFixed(1)}%</div>
              </div>
                <div className="bg-purple-50 rounded-lg p-4 border-l-4 border-purple-500">
                  <div className="text-sm text-purple-600 font-medium">Latest Analysis</div>
                  <div className="text-lg font-bold text-purple-900 mt-1">
                    {stats.latest_date ? new Date(stats.latest_date).toLocaleString() : 'N/A'}
                  </div>
                </div>
              </div>
            )}
          </div>
        </header>

        {/* Content Area */}
        <div className="p-8 overflow-y-auto h-[calc(100vh-200px)]">
        {/* Search and Filters */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <div className="flex flex-col gap-4">
            {/* Search Bar */}
            <div className="flex gap-4">
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                <input
                  type="text"
                  placeholder="Search by name, location, or diagnosis..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleApplyFilters()}
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
              <button
                onClick={() => setShowFilters(!showFilters)}
                className="flex items-center gap-2 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
              >
                <Filter className="w-4 h-4" />
                Filters
                {showFilters ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </button>
              <button
                onClick={handleApplyFilters}
                className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Search
              </button>
            </div>

            {/* Advanced Filters */}
            {showFilters && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4 border-t">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Start Date
                  </label>
                  <input
                    type="date"
                    value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    End Date
                  </label>
                  <input
                    type="date"
                    value={endDate}
                    onChange={(e) => setEndDate(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Diagnosis
                  </label>
                  <select
                    value={selectedDiagnosis}
                    onChange={(e) => setSelectedDiagnosis(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">All Diagnoses</option>
                    {uniqueDiagnoses.map(diagnosis => (
                      <option key={diagnosis} value={diagnosis}>{diagnosis}</option>
                    ))}
                  </select>
                </div>
                <div className="md:col-span-3 flex justify-end">
                  <button
                    onClick={handleResetFilters}
                    className="px-4 py-2 text-gray-600 hover:text-gray-800"
                  >
                    Reset Filters
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Results Table */}
        <div className="bg-white rounded-lg shadow overflow-hidden">
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Loader className="w-8 h-8 animate-spin text-blue-600" />
              <span className="ml-3 text-gray-600">Loading history...</span>
            </div>
          ) : error ? (
            <div className="flex items-center justify-center py-12 text-red-600">
              <AlertCircle className="w-6 h-6 mr-2" />
              {error}
            </div>
          ) : history.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-gray-500">
              <FileText className="w-12 h-12 mb-3 text-gray-400" />
              <p className="text-lg font-medium">No records found</p>
              <p className="text-sm">Try adjusting your search or filters</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Patient Name
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Email
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Age
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Gender
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Location
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Diagnosis
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Confidence
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Date & Time
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {history.map((record) => (
                    <tr key={record.id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm font-medium text-gray-900">{record.patient_name}</div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-900">{record.patient_email || 'N/A'}</div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-900">{record.age}</div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-900 capitalize">{record.gender}</div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-900">{record.location || 'N/A'}</div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${getRiskColor(record.diagnosis)}`}>
                          {record.diagnosis}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-900">{record.confidence.toFixed(1)}%</div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-900">
                          {record.created_at ? new Date(record.created_at).toLocaleString() : 'N/A'}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                        <button
                          onClick={() => handleViewDetails(record.id)}
                          className="text-blue-600 hover:text-blue-900 flex items-center gap-1"
                        >
                          <Eye className="w-4 h-4" />
                          View
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Results Count */}
        {!loading && !error && history.length > 0 && (
          <div className="mt-4 text-sm text-gray-600 text-center">
            Showing {history.length} record{history.length !== 1 ? 's' : ''}
          </div>
        )}
        </div>
      </div>

      {/* Detail Modal */}
      {showDetailModal && selectedRecord && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex justify-between items-start mb-6">
                <h2 className="text-2xl font-bold text-gray-900">Analysis Details</h2>
                <button
                  onClick={() => setShowDetailModal(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Patient Information */}
                <div className="bg-gray-50 rounded-lg p-4">
                  <h3 className="font-semibold text-gray-900 mb-3">Patient Information</h3>
                  <div className="space-y-2 text-sm">
                    <div><span className="font-medium">Name:</span> {selectedRecord.patient_name}</div>
                    <div><span className="font-medium">Age:</span> {selectedRecord.age}</div>
                    <div><span className="font-medium">Gender:</span> {selectedRecord.gender}</div>
                    <div><span className="font-medium">Location:</span> {selectedRecord.location || 'N/A'}</div>
                    <div><span className="font-medium">Date:</span> {selectedRecord.created_at ? new Date(selectedRecord.created_at).toLocaleString() : 'N/A'}</div>
                  </div>
                </div>

                {/* Diagnosis Information */}
                <div className="bg-gray-50 rounded-lg p-4">
                  <h3 className="font-semibold text-gray-900 mb-3">Diagnosis</h3>
                  <div className="space-y-2 text-sm">
                    <div>
                      <span className="font-medium">Result:</span> 
                      <span className={`ml-2 px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${getRiskColor(selectedRecord.diagnosis)}`}>
                        {selectedRecord.diagnosis}
                      </span>
                    </div>
                    <div><span className="font-medium">Confidence:</span> {selectedRecord.confidence.toFixed(2)}%</div>
                  </div>
                </div>

                {/* Images */}
                {selectedRecord.image_path && (
                  <div className="md:col-span-2">
                    <h3 className="font-semibold text-gray-900 mb-3">Images</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <p className="text-sm text-gray-600 mb-2">Original Image</p>
                        <img 
                          src={`http://localhost:5001${selectedRecord.image_path}`} 
                          alt="Original" 
                          className="w-full rounded-lg border"
                        />
                      </div>
                      {selectedRecord.viz_path && (
                        <div>
                          <p className="text-sm text-gray-600 mb-2">Visualization</p>
                          <img 
                            src={`http://localhost:5001${selectedRecord.viz_path}`} 
                            alt="Visualization" 
                            className="w-full rounded-lg border"
                          />
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* All Predictions */}
                {selectedRecord.all_predictions && selectedRecord.all_predictions.length > 0 && (
                  <div className="md:col-span-2">
                    <h3 className="font-semibold text-gray-900 mb-3">All Predictions</h3>
                    <div className="space-y-2">
                      {selectedRecord.all_predictions.map((pred, idx) => (
                        <div key={idx} className="flex justify-between items-center bg-gray-50 rounded p-2">
                          <span className="text-sm">{pred.class}</span>
                          <span className="text-sm font-medium">{pred.confidence.toFixed(2)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* LLM Advice */}
                {selectedRecord.llm_advice && (
                  <div className="md:col-span-2">
                    <h3 className="font-semibold text-gray-900 mb-3">Medical Insights</h3>
                    <div 
                      className="bg-blue-50 rounded-lg p-4 text-sm text-gray-700"
                      dangerouslySetInnerHTML={{ __html: formatMarkdown(selectedRecord.llm_advice) }}
                    />
                  </div>
                )}

                {/* Report Download */}
                {selectedRecord.report_path && (
                  <div className="md:col-span-2">
                    <a
                      href={`http://localhost:5001${selectedRecord.report_path}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                    >
                      <Download className="w-4 h-4" />
                      Download Full Report (PDF)
                    </a>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

