import HomeChild from "../../components/HomeChild.jsx";
import Button from "../../components/Button.jsx";
import Heading1 from "../../components/headings/Heading1.jsx";
import {useNavigate} from "react-router-dom";
import {useState} from "react";

function Home() {
    const navigate = useNavigate();
    const [showHowItWorks, setShowHowItWorks] = useState(false);

    return (
        <HomeChild className="flex flex-col gap-8 items-center justify-center text-center">
            <Heading1 text="Take your Skin Cancer Test Now!" />
            <p className="text-xl">
                Skin cancers are cancers that arise from the skin. They are due to the development of abnormal cells in the skin. <br /> Machine learning and neural networks can analyze images of moles and lesions to identify suspicious patterns, aiding in early detection.
            </p>
            <div className="flex justify-center gap-5 items-center">
                <Button text="Take a test" onClick={() => navigate("/take-a-test")}/>
                <Button text="How it works?" filled={false} onClick={() => setShowHowItWorks(true)}/>
            </div>

            {/* How It Works Modal */}
            {showHowItWorks && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
                    <div className="bg-white rounded-lg max-w-2xl w-full max-h-[80vh] overflow-y-auto p-6 relative">
                        <button 
                            onClick={() => setShowHowItWorks(false)}
                            className="absolute top-4 right-4 text-gray-500 hover:text-gray-700 text-2xl font-bold"
                        >
                            ×
                        </button>
                        
                        <h2 className="text-3xl font-bold mb-6 text-center">How It Works</h2>
                        
                        <div className="space-y-6">
                            <div className="flex items-start space-x-4">
                                <div className="bg-black text-white rounded-full w-8 h-8 flex items-center justify-center font-bold flex-shrink-0">
                                    1
                                </div>
                                <div>
                                    <h3 className="text-xl font-semibold mb-2">Upload Your Image</h3>
                                    <p className="text-gray-700">
                                        Take a clear, well-lit photo of your skin concern (mole, lesion, or spot). 
                                        Make sure the image is in focus and the area is clearly visible.
                                    </p>
                                </div>
                            </div>

                            <div className="flex items-start space-x-4">
                                <div className="bg-black text-white rounded-full w-8 h-8 flex items-center justify-center font-bold flex-shrink-0">
                                    2
                                </div>
                                <div>
                                    <h3 className="text-xl font-semibold mb-2">AI Analysis</h3>
                                    <p className="text-gray-700">
                                        Our advanced machine learning model analyzes the image using deep neural networks 
                                        trained on thousands of dermatological images to identify patterns and features.
                                    </p>
                                </div>
                            </div>

                            <div className="flex items-start space-x-4">
                                <div className="bg-black text-white rounded-full w-8 h-8 flex items-center justify-center font-bold flex-shrink-0">
                                    3
                                </div>
                                <div>
                                    <h3 className="text-xl font-semibold mb-2">Get Results</h3>
                                    <p className="text-gray-700">
                                        Receive an instant assessment with confidence levels for different skin conditions. 
                                        The system can identify various types including melanoma, nevus, and other skin lesions.
                                    </p>
                                </div>
                            </div>

                            <div className="flex items-start space-x-4">
                                <div className="bg-black text-white rounded-full w-8 h-8 flex items-center justify-center font-bold flex-shrink-0">
                                    4
                                </div>
                                <div>
                                    <h3 className="text-xl font-semibold mb-2">Professional Consultation</h3>
                                    <p className="text-gray-700">
                                        <strong>Important:</strong> This tool is for screening purposes only. 
                                        Always consult with a dermatologist or healthcare professional for proper diagnosis and treatment.
                                    </p>
                                </div>
                            </div>
                        </div>

                        <div className="mt-8 p-4 bg-yellow-50 border-l-4 border-yellow-400">
                            <div className="flex">
                                <div className="flex-shrink-0">
                                    <svg className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                                        <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                                    </svg>
                                </div>
                                <div className="ml-3">
                                    <p className="text-sm text-yellow-700">
                                        <strong>Medical Disclaimer:</strong> This AI screening tool is not a substitute for professional medical advice, diagnosis, or treatment. Early detection is crucial for skin cancer treatment success.
                                    </p>
                                </div>
                            </div>
                        </div>

                        <div className="mt-6 text-center">
                            <Button text="Get Started" onClick={() => {setShowHowItWorks(false); navigate("/take-a-test");}}/>
                        </div>
                    </div>
                </div>
            )}
        </HomeChild>
    )
}

export default Home;