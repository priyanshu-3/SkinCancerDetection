import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  CheckCircle,
  AlertTriangle,
  Info,
  TrendingUp,
  Brain,
} from 'lucide-react'

export default function Results() {
  const navigate = useNavigate()
  const [result, setResult] = useState(null)

  useEffect(() => {
    const stored = sessionStorage.getItem('analysisResult')
    if (stored) {
      setResult(JSON.parse(stored))
    } else {
      navigate('/analysis')
    }
  }, [navigate])

  if (!result) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  const { prediction, ensemble, llm_insights, explainability } = result

  // Determine risk level
  const getRiskLevel = (className) => {
    const highRisk = ['Melanoma', 'Basal cell carcinoma']
    const mediumRisk = ['Actinic keratoses']
    if (highRisk.includes(className)) return 'high'
    if (mediumRisk.includes(className)) return 'medium'
    return 'low'
  }

  const riskLevel = getRiskLevel(prediction.class)

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

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      {/* Header */}
      <div className="mb-8">
        <button
          onClick={() => navigate('/analysis')}
          className="text-primary-600 hover:text-primary-700 mb-4"
        >
          ← New Analysis
        </button>
        <h1 className="text-3xl font-bold">Analysis Results</h1>
      </div>

      <div className="grid lg:grid-cols-3 gap-8">
        {/* Main Results Column */}
        <div className="lg:col-span-2 space-y-6">
          {/* Primary Prediction */}
          <div className="card">
            <h2 className="text-xl font-semibold mb-4">Primary Prediction</h2>
            <div className="flex items-start justify-between">
              <div>
                <div className="text-3xl font-bold text-gray-900 mb-2">
                  {prediction.class}
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-lg text-gray-600">Confidence:</span>
                  <span className="text-lg font-semibold text-primary-600">
                    {prediction.confidence.toFixed(1)}%
                  </span>
                </div>
              </div>
              <RiskBadge level={riskLevel} />
            </div>
          </div>

          {/* Ensemble Results */}
          {ensemble && (
            <div className="card">
              <h2 className="text-xl font-semibold mb-4 flex items-center">
                <Brain className="w-5 h-5 mr-2" />
                Ensemble Analysis
              </h2>
              <div className="grid grid-cols-2 gap-4">
                <MetricCard
                  label="Ensemble Confidence"
                  value={`${ensemble.confidence.toFixed(1)}%`}
                  subValue={`± ${ensemble.confidence_std.toFixed(1)}%`}
                />
                <MetricCard
                  label="Uncertainty Score"
                  value={ensemble.uncertainty_score.toFixed(3)}
                  subValue={`Agreement: ${ensemble.agreement_rate.toFixed(1)}%`}
                />
              </div>
              <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                <p className="text-sm text-blue-900">
                  {ensemble.recommendation}
                </p>
              </div>
            </div>
          )}

          {/* All Model Predictions */}
          {prediction.all_predictions && (
            <div className="card">
              <h2 className="text-xl font-semibold mb-4">
                All Model Predictions
              </h2>
              <div className="space-y-3">
                {Object.entries(prediction.all_predictions).map(
                  ([model, preds]) => (
                    <ModelPrediction key={model} model={model} predictions={preds} />
                  )
                )}
              </div>
            </div>
          )}

          {/* Explainability */}
          {explainability && (
            <div className="card">
              <h2 className="text-xl font-semibold mb-4">
                Explainability Visualizations
              </h2>
              <div className="grid md:grid-cols-2 gap-4">
                {explainability.saliency && (
                  <div>
                    <h3 className="font-medium mb-2">Saliency Map</h3>
                    <img
                      src={explainability.saliency}
                      alt="Saliency"
                      className="rounded-lg border"
                    />
                    <p className="text-xs text-gray-600 mt-1">
                      Highlights important regions for classification
                    </p>
                  </div>
                )}
                {explainability.lime && (
                  <div>
                    <h3 className="font-medium mb-2">LIME Explanation</h3>
                    <img
                      src={explainability.lime}
                      alt="LIME"
                      className="rounded-lg border"
                    />
                    <p className="text-xs text-gray-600 mt-1">
                      Local interpretable model-agnostic explanations
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* LLM Insights */}
          {llm_insights && (
            <div className="card">
              <h2 className="text-lg font-semibold mb-3 flex items-center">
                <TrendingUp className="w-5 h-5 mr-2" />
                AI Insights
              </h2>
              <div className="prose prose-sm max-w-none">
                <div
                  dangerouslySetInnerHTML={{
                    __html: formatMarkdown(llm_insights),
                  }}
                />
              </div>
            </div>
          )}

          {/* Recommendations */}
          <div className="card">
            <h2 className="text-lg font-semibold mb-3">Recommendations</h2>
            <div className="space-y-2">
              {riskLevel === 'high' && (
                <RecommendationItem
                  icon={<AlertTriangle className="w-5 h-5" />}
                  text="Consult a dermatologist immediately"
                  type="warning"
                />
              )}
              {riskLevel === 'medium' && (
                <RecommendationItem
                  icon={<Info className="w-5 h-5" />}
                  text="Schedule a dermatologist appointment"
                  type="info"
                />
              )}
              {riskLevel === 'low' && (
                <RecommendationItem
                  icon={<CheckCircle className="w-5 h-5" />}
                  text="Continue regular monitoring"
                  type="success"
                />
              )}
              <RecommendationItem
                icon={<Info className="w-5 h-5" />}
                text="Monitor for changes in size, color, or shape"
                type="info"
              />
            </div>
          </div>

          {/* Disclaimer */}
          <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <p className="text-sm text-yellow-800">
              <strong>⚠️ Medical Disclaimer:</strong> This is an AI tool for
              informational purposes only. Always consult qualified healthcare
              professionals.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

function RiskBadge({ level }) {
  const configs = {
    high: {
      bg: 'bg-red-100',
      text: 'text-red-800',
      border: 'border-red-300',
      label: 'High Risk',
    },
    medium: {
      bg: 'bg-yellow-100',
      text: 'text-yellow-800',
      border: 'border-yellow-300',
      label: 'Medium Risk',
    },
    low: {
      bg: 'bg-green-100',
      text: 'text-green-800',
      border: 'border-green-300',
      label: 'Low Risk',
    },
  }

  const config = configs[level]

  return (
    <div
      className={`px-4 py-2 rounded-full border ${config.bg} ${config.text} ${config.border} font-semibold`}
    >
      {config.label}
    </div>
  )
}

function MetricCard({ label, value, subValue }) {
  return (
    <div className="p-4 bg-gray-50 rounded-lg">
      <div className="text-sm text-gray-600 mb-1">{label}</div>
      <div className="text-2xl font-bold text-gray-900">{value}</div>
      {subValue && <div className="text-sm text-gray-500 mt-1">{subValue}</div>}
    </div>
  )
}

function ModelPrediction({ model, predictions }) {
  const topPred = Object.entries(predictions).sort((a, b) => b[1] - a[1])[0]
  const [className, confidence] = topPred

  return (
    <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
      <div>
        <div className="font-medium">{model}</div>
        <div className="text-sm text-gray-600">{className}</div>
      </div>
      <div className="text-right">
        <div className="font-semibold text-primary-600">
          {(confidence * 100).toFixed(1)}%
        </div>
      </div>
    </div>
  )
}

function RecommendationItem({ icon, text, type }) {
  const colors = {
    warning: 'text-red-600',
    info: 'text-blue-600',
    success: 'text-green-600',
  }

  return (
    <div className="flex items-start space-x-2">
      <div className={colors[type]}>{icon}</div>
      <p className="text-sm text-gray-700">{text}</p>
    </div>
  )
}

