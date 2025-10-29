import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, Download, Calendar, User, MapPin, Activity } from 'lucide-react'

export default function AnalysisDetail() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [analysis, setAnalysis] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

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

  useEffect(() => {
    const fetchAnalysis = async () => {
      try {
        setLoading(true)
        console.log('Fetching analysis for ID:', id)
        
        // Get patient email from localStorage if available
        const patientData = JSON.parse(localStorage.getItem('patient') || '{}')
        const patientEmail = patientData.email
        
        // Build URL with patient_email parameter if available
        let url = `http://localhost:5001/api/history/${id}`
        if (patientEmail) {
          url += `?patient_email=${encodeURIComponent(patientEmail)}`
        }
        
        const response = await fetch(url, {
          credentials: 'include'
        })
        
        console.log('Response status:', response.status)
        console.log('Response ok:', response.ok)
        
        if (response.ok) {
          const data = await response.json()
          console.log('Response data:', data)
          setAnalysis(data.analysis)
        } else {
          const errorText = await response.text()
          console.log('Error response:', errorText)
          setError('Analysis not found')
        }
      } catch (err) {
        setError('Failed to load analysis details')
        console.error('Error fetching analysis:', err)
      } finally {
        setLoading(false)
      }
    }

    if (id) {
      fetchAnalysis()
    }
  }, [id])

  const handleDownloadReport = () => {
    if (analysis?.report_path) {
      window.open(`http://localhost:5001${analysis.report_path}`, '_blank')
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading analysis details...</p>
        </div>
      </div>
    )
  }

  if (error || !analysis) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="text-red-500 text-6xl mb-4">⚠️</div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Analysis Not Found</h2>
          <p className="text-gray-600 mb-4">{error || 'The requested analysis could not be found.'}</p>
          <button
            onClick={() => navigate(-1)}
            className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors"
          >
            Go Back
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-4xl mx-auto p-6">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <button
              onClick={() => navigate(-1)}
              className="flex items-center text-gray-600 hover:text-gray-900 transition-colors"
            >
              <ArrowLeft className="w-5 h-5 mr-2" />
              Back
            </button>
            <button
              onClick={handleDownloadReport}
              className="flex items-center bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
            >
              <Download className="w-4 h-4 mr-2" />
              Download Report
            </button>
          </div>
          
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Analysis Details</h1>
          <div className="flex items-center space-x-4 text-sm text-gray-600">
            <div className="flex items-center">
              <Calendar className="w-4 h-4 mr-1" />
              {analysis.created_at}
            </div>
            <div className="flex items-center">
              <User className="w-4 h-4 mr-1" />
              {analysis.patient_name}
            </div>
            <div className="flex items-center">
              <MapPin className="w-4 h-4 mr-1" />
              {analysis.location}
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Diagnosis Card */}
          <div className="bg-white rounded-lg shadow-sm p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
              <Activity className="w-5 h-5 mr-2 text-blue-600" />
              Diagnosis
            </h2>
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-600 mb-2">
                {analysis.diagnosis}
              </div>
              <div className="text-lg text-gray-600">
                Confidence: {analysis.confidence}%
              </div>
            </div>
          </div>

          {/* Patient Info Card */}
          <div className="bg-white rounded-lg shadow-sm p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Patient Information</h2>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">Name:</span>
                <span className="font-medium">{analysis.patient_name}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Age:</span>
                <span className="font-medium">{analysis.age}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Gender:</span>
                <span className="font-medium">{analysis.gender}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Location:</span>
                <span className="font-medium">{analysis.location}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Image and Visualization */}
        {(analysis.image_path || analysis.viz_path) && (
          <div className="bg-white rounded-lg shadow-sm p-6 mt-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Images</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {analysis.image_path && (
                <div>
                  <h3 className="text-lg font-medium text-gray-700 mb-2">Original Image</h3>
                  <img
                    src={`http://localhost:5001${analysis.image_path}`}
                    alt="Analysis image"
                    className="w-full h-64 object-cover rounded-lg border"
                  />
                </div>
              )}
              {analysis.viz_path && (
                <div>
                  <h3 className="text-lg font-medium text-gray-700 mb-2">Visualization</h3>
                  <img
                    src={`http://localhost:5001${analysis.viz_path}`}
                    alt="Analysis visualization"
                    className="w-full h-64 object-cover rounded-lg border"
                  />
                </div>
              )}
            </div>
          </div>
        )}

        {/* Medical Advice */}
        {analysis.llm_advice && (
          <div className="bg-white rounded-lg shadow-sm p-6 mt-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Medical Advice</h2>
            <div className="prose max-w-none">
              <div 
                className="text-gray-700 leading-relaxed"
                dangerouslySetInnerHTML={{ 
                  __html: formatMarkdown(analysis.llm_advice)
                }}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
