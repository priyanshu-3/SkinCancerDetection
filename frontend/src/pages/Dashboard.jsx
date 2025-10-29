import { useState, useRef } from 'react'

export default function Dashboard() {
  const [currentImageFile, setCurrentImageFile] = useState(null)
  const [currentImageData, setCurrentImageData] = useState(null)
  const [previewImage, setPreviewImage] = useState(null)
  const [capturedImage, setCapturedImage] = useState(null)
  const [stream, setStream] = useState(null)
  const [cameraActive, setCameraActive] = useState(false)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [errorModal, setErrorModal] = useState({ show: false, title: '', message: '' })
  const [activeTab, setActiveTab] = useState('upload')
  
  const [patientInfo, setPatientInfo] = useState({
    name: '',
    age: '',
    gender: '',
    location: '',
    postal_code: '',
  })

  const [gpsLoading, setGpsLoading] = useState(false)
  const [detectedLocation, setDetectedLocation] = useState('')
  const [lastCoords, setLastCoords] = useState(null)
  const [rawReverse, setRawReverse] = useState(null)
  const [showGpsDebug, setShowGpsDebug] = useState(false)

  const fileInputRef = useRef(null)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)

  const showErrorModal = (title, message) => {
    setErrorModal({ show: true, title, message })
  }

  const closeErrorModal = () => {
    setErrorModal({ show: false, title: '', message: '' })
  }

  const handleFileSelect = (file) => {
    if (!file.type.startsWith('image/')) {
      showErrorModal('Invalid File Type', 'Please select an image file (JPG, PNG, JPEG, etc.).')
      return
    }

    setCurrentImageFile(file)
    const reader = new FileReader()
    reader.onload = (e) => {
      setPreviewImage(e.target.result)
    }
    reader.readAsDataURL(file)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    const files = e.dataTransfer.files
    if (files.length > 0) {
      handleFileSelect(files[0])
    }
  }

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      })
      setStream(mediaStream)
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream
      }
      setCameraActive(true)
    } catch (err) {
      showErrorModal(
        'Camera Access Error',
        `Unable to access camera: ${err.message}\n\nPlease check camera permissions in your browser settings.`
      )
    }
  }

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop())
      setStream(null)
      setCameraActive(false)
      setCapturedImage(null)
    }
  }

  const captureImage = () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (video && canvas) {
      const ctx = canvas.getContext('2d')
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      ctx.drawImage(video, 0, 0)
      const imageData = canvas.toDataURL('image/jpeg', 0.8)
      setCurrentImageData(imageData)
      setCapturedImage(imageData)
    }
  }

  const performAnalysis = async (imageFile) => {
    const { name, age, gender, location, postal_code } = patientInfo

    if (!name || !age || !gender) {
      showErrorModal(
        'Required Fields Missing',
        'Please fill in all required fields:\n• Name\n• Age\n• Gender'
      )
      return
    }
    
    // Validate age: must be a number and less than 100
    const numericAge = Number(age)
    if (Number.isNaN(numericAge) || numericAge <= 0 || numericAge >= 100) {
      showErrorModal('Invalid Age', 'Please enter the correct age (less than 100).')
      return
    }

    setLoading(true)
    setResults(null)

    try {
      const formData = new FormData()
      formData.append('image', imageFile)
      formData.append('name', name)
      formData.append('age', age)
      formData.append('gender', gender)
      formData.append('location', location)
  // include postal/pin code if present
  formData.append('postal_code', postal_code || '')

      const response = await fetch('http://localhost:5001/analyze', {
        method: 'POST',
        body: formData,
        credentials: 'include',
      })

      const result = await response.json()

      if (!response.ok || result.error) {
        const errorTitle = result.error || 'Analysis Failed'
        const errorMessage = result.message || 'Unable to analyze the image. Please try again.'
        showErrorModal(errorTitle, errorMessage)
        return
      }

      setResults(result)
      setTimeout(() => {
        document.getElementById('results')?.scrollIntoView({ behavior: 'smooth' })
      }, 100)
    } catch (error) {
      showErrorModal(
        'Analysis Failed',
        `An unexpected error occurred:\n\n${error.message}\n\nPlease try again or contact support if the problem persists.`
      )
    } finally {
      setLoading(false)
    }
  }

  const analyzeImage = () => {
    if (!currentImageFile) {
      showErrorModal('No Image Selected', 'Please select or upload an image before analyzing.')
      return
    }
    performAnalysis(currentImageFile)
  }

  const analyzeCapturedImage = async () => {
    if (!currentImageData) {
      showErrorModal('No Image Captured', 'Please capture an image using the camera before analyzing.')
      return
    }

    const response = await fetch(currentImageData)
    const blob = await response.blob()
    const file = new File([blob], 'captured.jpg', { type: 'image/jpeg' })
    performAnalysis(file)
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

  // Use browser geolocation + reverse geocoding (OpenStreetMap Nominatim)
  const useGPS = async () => {
    setGpsLoading(true)
    setDetectedLocation('')
    try {
      if (!navigator.geolocation) {
        showErrorModal('Geolocation Unsupported', 'Your browser does not support geolocation.')
        setGpsLoading(false)
        return
      }

      navigator.geolocation.getCurrentPosition(
        async (position) => {
          const lat = position.coords.latitude
          const lon = position.coords.longitude
          setLastCoords({ lat, lon, accuracy: position.coords.accuracy })

          try {
            // Reverse geocode using OpenStreetMap Nominatim
            const resp = await fetch(
              `https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${lat}&lon=${lon}&addressdetails=1&accept-language=en`,
              { headers: { Accept: 'application/json' } }
            )
            const data = await resp.json()
            setRawReverse(data)
            const addr = data.address || {}

            // Prefer the most useful locality fields (city/town/village). If the result is a small
            // hamlet/suburb (e.g. Chikkajala) but a larger nearby town is present in display_name
            // (e.g. Devanahalli), prefer the larger town for user-friendliness.
            const primaryLocality = addr.city || addr.town || addr.village || addr.municipality || addr.county || ''
            const smallLocality = addr.hamlet || addr.suburb || addr.neighbourhood || addr.road || ''
            const state = addr.state || addr.state_district || addr.region || ''
            const country = addr.country || ''

            // Build candidate parts
            const parts = []
            if (primaryLocality) parts.push(primaryLocality)
            if (state && state !== primaryLocality) parts.push(state)
            if (country) parts.push(country)

            let locText = parts.filter(Boolean).join(', ')

            // Heuristic: if primaryLocality is missing or is a small locality (hamlet/suburb),
            // try to pick the next component from display_name (often the larger town)
            if ((!primaryLocality || smallLocality) && data.display_name) {
              const comps = data.display_name.split(',').map(s => s.trim()).filter(Boolean)
              // comps[0] is the nearest name (may be hamlet). Try comps[1] or comps[2] as the town.
              if (comps.length >= 2) {
                // prefer comps[1] if it's different from comps[0]
                const candidate = comps[1]
                if (candidate && candidate !== comps[0]) {
                  // rebuild using candidate as locality
                  const candidateParts = [candidate]
                  if (comps.length >= 3) candidateParts.push(comps[2])
                  if (comps.length >= 4) candidateParts.push(comps[3])
                  locText = candidateParts.slice(0, 3).join(', ')
                }
              }
            }

            // Final fallback to display_name or lat/lon
            if (!locText && data.display_name) {
              locText = data.display_name.split(',').slice(0, 3).map(s => s.trim()).join(', ')
            }
            if (!locText) {
              locText = `${lat.toFixed(5)}, ${lon.toFixed(5)}`
            }

            setPatientInfo((p) => ({ ...p, location: locText }))
            setDetectedLocation(locText)
          } catch (err) {
            // fallback to lat,lon string
            const fallback = `${lat.toFixed(5)}, ${lon.toFixed(5)}`
            setPatientInfo((p) => ({ ...p, location: fallback }))
            setDetectedLocation(fallback)
          } finally {
            setGpsLoading(false)
          }
        },
        (err) => {
          setGpsLoading(false)
          showErrorModal('GPS Error', 'Unable to retrieve your location. Please enable location permissions and try again.')
        },
        { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
      )
    } catch (err) {
      setGpsLoading(false)
      showErrorModal('GPS Error', err.message)
    }
  }

  const clearDetected = () => {
    setDetectedLocation('')
    setPatientInfo((p) => ({ ...p, location: '' }))
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#667eea] to-[#764ba2] py-5">
      <div className="container mx-auto px-4">
        <div className="max-w-7xl mx-auto bg-white/95 backdrop-blur-sm rounded-3xl shadow-2xl p-8">
          {/* Header */}
          <div className="mb-8">
            <div className="flex justify-between items-start mb-4">
              <div className="flex-1 text-center">
                <h1 className="text-5xl font-bold text-blue-600 mb-3">
                  <i className="fas fa-microscope mr-3"></i>
                  Skin Cancer Detection System
                </h1>
                <p className="text-xl text-gray-600">
                  AI-powered skin lesion analysis with personalized medical insights
                </p>
              </div>
              <a
                href="/history"
                className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors shadow-md"
              >
                <i className="fas fa-history"></i>
                View History
              </a>
            </div>
          </div>

          {/* Patient Information */}
          <div className="card bg-white shadow-lg rounded-xl p-6 mb-6">
            <div className="flex items-center mb-4 p-3 bg-blue-600 text-white rounded-t-xl -mx-6 -mt-6 mb-4">
              <i className="fas fa-user text-xl mr-3"></i>
              <h5 className="text-xl font-semibold">Patient Information</h5>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">Full Name *</label>
                <input
                  type="text"
                  value={patientInfo.name}
                  onChange={(e) => setPatientInfo({ ...patientInfo, name: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Age *</label>
                <input
                  type="number"
                  value={patientInfo.age}
                  onChange={(e) => setPatientInfo({ ...patientInfo, age: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  min="1"
                  max="99"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Gender *</label>
                <select
                  value={patientInfo.gender}
                  onChange={(e) => setPatientInfo({ ...patientInfo, gender: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                >
                  <option value="">Select Gender</option>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Location (City)</label>
                <div className="relative">
                  <input
                    type="text"
                    value={patientInfo.location}
                    onChange={(e) => setPatientInfo({ ...patientInfo, location: e.target.value })}
                    className="w-full pr-10 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="Enter city or use GPS"
                  />
                  <button
                    type="button"
                    onClick={useGPS}
                    disabled={gpsLoading}
                    title="Use GPS to detect location"
                    className="absolute right-1 top-1/2 transform -translate-y-1/2 p-2 bg-white border border-gray-200 rounded-md text-blue-600 hover:bg-blue-50 disabled:opacity-60 disabled:cursor-not-allowed"
                    aria-label="Use GPS"
                  >
                    {gpsLoading ? (
                      <i className="fas fa-spinner animate-spin"></i>
                    ) : (
                      <i className="fas fa-map-marker-alt"></i>
                    )}
                  </button>
                </div>
                {detectedLocation && (
                  <p className="text-green-600 text-sm mt-2">
                    Location detected: {detectedLocation}
                    <button onClick={clearDetected} className="ml-2 text-blue-600 underline text-xs">
                      Clear
                    </button>
                  </p>
                )}
                <small className="text-gray-500 block mt-1">Click GPS button to auto-detect location</small>

                {/* Postal / PIN code (optional) */}
                <div className="mt-3">
                  <label className="block text-sm font-medium mb-1">PIN / Postal Code (optional)</label>
                  <input
                    type="text"
                    value={patientInfo.postal_code}
                    onChange={(e) => setPatientInfo({ ...patientInfo, postal_code: e.target.value })}
                    className="w-1/2 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="Pin code"
                  />
                </div>

                {/* GPS debug toggle & panel */}
                <div className="mt-2">
                  <button
                    type="button"
                    onClick={() => setShowGpsDebug((s) => !s)}
                    className="text-xs text-gray-600 hover:underline"
                  >
                    {showGpsDebug ? 'Hide GPS debug' : 'Show GPS debug'}
                  </button>
                  {showGpsDebug && (
                    <div className="mt-2 p-3 bg-gray-100 rounded-md text-xs text-gray-800">
                      <div>
                        <strong>Raw Coords:</strong>
                        <div>{lastCoords ? `${lastCoords.lat.toFixed(6)}, ${lastCoords.lon ? lastCoords.lon.toFixed(6) : ''} (acc ${lastCoords.accuracy}m)` : 'N/A'}</div>
                      </div>
                      <div className="mt-2">
                        <strong>display_name:</strong>
                        <div>{rawReverse?.display_name || 'N/A'}</div>
                      </div>
                      <div className="mt-2">
                        <strong>address (compact):</strong>
                        <pre className="text-xs max-h-32 overflow-auto bg-white p-2 rounded">{rawReverse ? JSON.stringify(rawReverse.address || rawReverse, null, 2) : 'N/A'}</pre>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Image Upload Section */}
          <div className="card bg-white shadow-lg rounded-xl p-6 mb-6">
            <div className="flex items-center mb-4 p-3 bg-green-600 text-white rounded-t-xl -mx-6 -mt-6 mb-4">
              <i className="fas fa-camera text-xl mr-3"></i>
              <h5 className="text-xl font-semibold">Image Capture & Upload</h5>
            </div>

            {/* Tabs */}
            <div className="flex border-b mb-4">
              <button
                onClick={() => setActiveTab('upload')}
                className={`px-6 py-2 font-semibold rounded-t-lg ${
                  activeTab === 'upload'
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                File Upload
              </button>
              <button
                onClick={() => setActiveTab('camera')}
                className={`px-6 py-2 font-semibold rounded-t-lg ${
                  activeTab === 'camera'
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                Camera Capture
              </button>
            </div>

            {/* Upload Tab */}
            {activeTab === 'upload' && (
              <div>
                <div
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className="border-3 border-dashed border-blue-500 rounded-xl p-10 text-center bg-gray-50 hover:bg-blue-50 hover:scale-[1.02] transition-all cursor-pointer"
                >
                  <i className="fas fa-cloud-upload-alt text-6xl text-blue-600 mb-4"></i>
                  <h4 className="text-xl font-semibold mb-2">Drag & Drop Your Image Here</h4>
                  <p className="text-gray-600">
                    or <span className="text-blue-600 underline">browse files</span>
                  </p>
                  <small className="text-gray-500 block mt-2">
                    Supported formats: JPG, PNG, JPEG (Max: 16MB)
                  </small>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
                    className="hidden"
                  />
                </div>
                {previewImage && (
                  <div className="text-center mt-6">
                    <img
                      src={previewImage}
                      alt="Preview"
                      className="max-w-xs max-h-64 mx-auto rounded-xl shadow-lg mb-4"
                    />
                    <button
                      onClick={analyzeImage}
                      className="px-6 py-3 bg-green-600 text-white rounded-full font-semibold hover:bg-green-700 hover:shadow-lg transition-all"
                    >
                      Analyze Image
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* Camera Tab */}
            {activeTab === 'camera' && (
              <div className="text-center">
                {!cameraActive ? (
                  <div className="border-3 border-dashed border-blue-500 rounded-xl p-10 bg-gray-50 max-w-md mx-auto">
                    <i className="fas fa-camera text-6xl text-blue-600 mb-4"></i>
                    <h5 className="text-xl font-semibold mb-4">Camera Access Required</h5>
                    <button
                      onClick={startCamera}
                      className="px-6 py-3 bg-gradient-to-r from-blue-600 to-blue-800 text-white rounded-full font-semibold hover:shadow-lg transition-all"
                    >
                      Start Camera
                    </button>
                  </div>
                ) : (
                  <div>
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      className="max-w-md mx-auto rounded-xl shadow-lg mb-4"
                    />
                    <canvas ref={canvasRef} className="hidden" />
                    <div className="flex justify-center gap-3 mb-4">
                      <button
                        onClick={captureImage}
                        className="px-6 py-3 bg-green-600 text-white rounded-full font-semibold hover:bg-green-700 transition-all"
                      >
                        <i className="fas fa-camera mr-2"></i>
                        Capture
                      </button>
                      <button
                        onClick={stopCamera}
                        className="px-6 py-3 bg-red-600 text-white rounded-full font-semibold hover:bg-red-700 transition-all"
                      >
                        <i className="fas fa-stop mr-2"></i>
                        Stop Camera
                      </button>
                    </div>
                  </div>
                )}
                {capturedImage && (
                  <div className="mt-6">
                    <img
                      src={capturedImage}
                      alt="Captured"
                      className="max-w-xs max-h-64 mx-auto rounded-xl shadow-lg mb-4"
                    />
                    <button
                      onClick={analyzeCapturedImage}
                      className="px-6 py-3 bg-green-600 text-white rounded-full font-semibold hover:bg-green-700 hover:shadow-lg transition-all"
                    >
                      Analyze Captured Image
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Loading */}
          {loading && (
            <div className="text-center py-12">
              <div className="inline-block w-16 h-16 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin mb-4"></div>
              <h5 className="text-xl font-semibold mb-2">Analyzing your image...</h5>
              <p className="text-gray-600">This may take a few moments</p>
            </div>
          )}

          {/* Results */}
          {results && (
            <div id="results" className="bg-white shadow-lg rounded-xl p-6 border-l-4 border-blue-600">
              <h3 className="text-2xl font-bold text-center mb-6">
                <i className="fas fa-chart-bar mr-2"></i>
                Analysis Results
              </h3>

              {/* PDF download button if available */}
              {results.report_path && (
                <div className="text-center mb-4">
                  <a
                    href={`http://localhost:5001${results.report_path}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-full hover:bg-blue-700"
                  >
                    <i className="fas fa-file-pdf mr-2"></i>
                    Download Report (PDF)
                  </a>
                </div>
              )}

              <div className="grid md:grid-cols-2 gap-6 mb-6">
                {/* Primary Diagnosis */}
                <div>
                  <h5 className="text-lg font-semibold mb-3">Primary Diagnosis</h5>
                  <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
                    <h4 className="text-2xl font-bold text-blue-600 mb-3">{results.prediction}</h4>
                    <div className="h-6 bg-gray-200 rounded-full overflow-hidden mb-2">
                      <div
                        className="h-full bg-gradient-to-r from-blue-600 to-blue-800 transition-all duration-1000"
                        style={{ width: `${results.confidence}%` }}
                      ></div>
                    </div>
                    <p className="mb-2">
                      <strong>Confidence:</strong> {results.confidence}%
                    </p>
                    <p className="mb-2">
                      <strong>Risk Level:</strong> {results.condition_info?.severity || 'Unknown'}
                    </p>
                    <p>
                      <strong>Description:</strong> {results.condition_info?.description || 'N/A'}
                    </p>
                  </div>
                </div>

                {/* All Predictions */}
                <div>
                  <h5 className="text-lg font-semibold mb-3">All Predictions</h5>
                  <div className="space-y-2">
                    {results.all_predictions && Object.entries(results.all_predictions)
                      .sort((a, b) => b[1].confidence - a[1].confidence)
                      .map(([condition, data], index) => (
                        <div
                          key={condition}
                          className="flex justify-between items-center p-3 bg-gray-50 rounded-lg border-l-4 border-blue-600"
                        >
                          <div>
                            <strong>{condition}</strong>
                            {index === 0 && (
                              <span className="ml-2 px-2 py-1 bg-blue-600 text-white text-xs rounded-full">
                                Top Prediction
                              </span>
                            )}
                          </div>
                          <span className="px-3 py-1 bg-gray-300 rounded-full text-sm font-semibold">
                            {(data.confidence * 100).toFixed(2)}%
                          </span>
                        </div>
                      ))}
                  </div>
                </div>
              </div>

              {/* Medical Insights */}
              {results.llm_advice && (
                <div className="mb-6">
                  <h5 className="text-lg font-semibold mb-3">Medical Insights & Recommendations</h5>
                  <div
                    className="bg-cyan-50 border-l-4 border-cyan-600 rounded-xl p-6"
                    dangerouslySetInnerHTML={{ __html: formatMarkdown(results.llm_advice) }}
                  />
                </div>
              )}

              {/* Images */}
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h5 className="text-lg font-semibold mb-3">Input Image</h5>
                  <img
                    src={`http://localhost:5001${results.image_path}`}
                    alt="Input"
                    className="w-full rounded-xl shadow-lg"
                  />
                </div>
                <div>
                  <h5 className="text-lg font-semibold mb-3">Analysis Visualization</h5>
                  <img
                    src={`http://localhost:5001${results.viz_path}`}
                    alt="Visualization"
                    className="w-full rounded-xl shadow-lg"
                  />
                </div>
              </div>
            </div>
          )}

          {/* Disclaimer */}
          <div className="bg-yellow-50 border border-yellow-300 rounded-xl p-4 mt-6">
            <h6 className="text-yellow-800 font-semibold mb-2">
              <i className="fas fa-exclamation-triangle mr-2"></i>
              Important Medical Disclaimer
            </h6>
            <p className="text-yellow-900 text-sm">
              This AI-powered tool is for educational and informational purposes only. The results
              provided are not a substitute for professional medical advice, diagnosis, or treatment.
              Always consult with qualified healthcare professionals for proper medical evaluation and
              care.
            </p>
          </div>
        </div>
      </div>

      {/* Error Modal */}
      {errorModal.show && (
        <div
          className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4 animate-fade-in"
          onClick={closeErrorModal}
        >
          <div
            className="bg-white rounded-2xl max-w-2xl w-full shadow-2xl animate-slide-down"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="bg-gradient-to-r from-red-500 to-red-600 text-white p-6 rounded-t-2xl flex items-center justify-between">
              <div className="flex items-center">
                <i className="fas fa-exclamation-triangle text-3xl mr-4"></i>
                <h4 className="text-2xl font-semibold">{errorModal.title}</h4>
              </div>
              <button
                onClick={closeErrorModal}
                className="text-white hover:text-gray-200 text-3xl leading-none"
              >
                &times;
              </button>
            </div>
            <div className="p-8">
              <p className="text-gray-700 text-lg whitespace-pre-line leading-relaxed">
                {errorModal.message}
              </p>
            </div>
            <div className="bg-gray-50 px-8 py-5 rounded-b-2xl text-right">
              <button
                onClick={closeErrorModal}
                className="px-8 py-3 bg-blue-600 text-white rounded-xl font-semibold hover:bg-blue-700 transition-all"
              >
                Try Again
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  )
}

