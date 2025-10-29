import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { Eye, EyeOff, User, Lock, Mail, AlertCircle, ArrowRight, Shield, Heart, Zap } from 'lucide-react'

export default function PatientLogin() {
  const navigate = useNavigate()
  const [formData, setFormData] = useState({
    username_or_email: '',
    password: ''
  })
  const [showPassword, setShowPassword] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    })
    setError('')
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')

    try {
      const response = await fetch('http://localhost:5001/api/patient/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
        credentials: 'include'
      })

      const data = await response.json()

      if (data.success) {
        // Store patient info in localStorage
        localStorage.setItem('patient', JSON.stringify(data.patient))
        localStorage.setItem('userType', 'patient')
        
        // Redirect to patient dashboard
        navigate('/patient-dashboard')
      } else {
        setError(data.error || 'Login failed')
      }
    } catch (err) {
      setError('Network error. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-400 via-purple-500 to-indigo-600 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full">
        {/* Main Login Card */}
        <div className="bg-white rounded-3xl shadow-2xl overflow-hidden">
          {/* Header Section with Gradient */}
          <div className="bg-gradient-to-r from-blue-500 to-purple-600 px-8 py-12 text-center">
            <div className="mx-auto w-16 h-16 bg-white/20 rounded-full flex items-center justify-center mb-4">
              <User className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-3xl font-bold text-white mb-2">Welcome Back</h1>
            <p className="text-white/90 text-sm mb-6">Sign in to access your personal health records</p>
            
            {/* Feature Tags */}
            <div className="flex justify-center space-x-3">
              <span className="bg-white/20 text-white text-xs px-3 py-1 rounded-full flex items-center gap-1">
                <Heart className="w-3 h-3" />
                Personal
              </span>
              <span className="bg-white/20 text-white text-xs px-3 py-1 rounded-full flex items-center gap-1">
                <Shield className="w-3 h-3" />
                Secure
              </span>
              <span className="bg-white/20 text-white text-xs px-3 py-1 rounded-full flex items-center gap-1">
                <Zap className="w-3 h-3" />
                Fast
              </span>
            </div>
          </div>

          {/* Form Section */}
          <div className="px-8 py-8">
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Email Field */}
              <div>
                <label htmlFor="username_or_email" className="block text-sm font-medium text-gray-700 mb-2">
                  Email Address
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <Mail className="h-5 w-5 text-gray-400" />
                  </div>
                  <input
                    id="username_or_email"
                    name="username_or_email"
                    type="email"
                    required
                    value={formData.username_or_email}
                    onChange={handleChange}
                    className="w-full pl-10 pr-3 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-900 placeholder-gray-500"
                    placeholder="your.email@example.com"
                  />
                </div>
              </div>

              {/* Password Field */}
              <div>
                <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-2">
                  Password
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <Lock className="h-5 w-5 text-gray-400" />
                  </div>
                  <input
                    id="password"
                    name="password"
                    type={showPassword ? 'text' : 'password'}
                    required
                    value={formData.password}
                    onChange={handleChange}
                    className="w-full pl-10 pr-10 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-900 placeholder-gray-500"
                    placeholder="Enter your password"
                  />
                  <button
                    type="button"
                    className="absolute inset-y-0 right-0 pr-3 flex items-center"
                    onClick={() => setShowPassword(!showPassword)}
                  >
                    {showPassword ? (
                      <EyeOff className="h-5 w-5 text-gray-400 hover:text-gray-600" />
                    ) : (
                      <Eye className="h-5 w-5 text-gray-400 hover:text-gray-600" />
                    )}
                  </button>
                </div>
              </div>

              {/* Error Message */}
              {error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                  <div className="flex">
                    <AlertCircle className="h-5 w-5 text-red-400 mt-0.5" />
                    <div className="ml-3">
                      <p className="text-sm text-red-800">{error}</p>
                    </div>
                  </div>
                </div>
              )}

              {/* Sign In Button */}
              <button
                type="submit"
                disabled={loading}
                className="w-full bg-gradient-to-r from-blue-500 to-purple-600 text-white py-3 px-4 rounded-lg font-medium hover:from-blue-600 hover:to-purple-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center gap-2"
              >
                {loading ? (
                  'Signing in...'
                ) : (
                  <>
                    Sign In
                    <ArrowRight className="w-4 h-4" />
                  </>
                )}
              </button>

              {/* Divider */}
              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-gray-300" />
                </div>
              </div>

              {/* Links */}
              <div className="space-y-4 text-center">
                <div>
                  <Link
                    to="/patient-register"
                    className="text-sm text-gray-600 hover:text-gray-800"
                  >
                    Don't have an account? <span className="text-blue-600 font-medium hover:text-blue-700">Create one now</span>
                  </Link>
                </div>
                
                <div>
                  <Link
                    to="/login"
                    className="text-sm text-gray-500 hover:text-gray-700"
                  >
                    Are you a doctor? <span className="text-green-600 font-medium hover:text-green-700">Admin Login</span>
                  </Link>
                </div>
              </div>
            </form>
          </div>


          {/* Footer */}
          <div className="px-8 py-4 bg-gray-50 text-center">
            <div className="flex items-center justify-center gap-2 text-xs text-gray-500">
              <Shield className="w-3 h-3" />
              Your data is encrypted and secure
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
