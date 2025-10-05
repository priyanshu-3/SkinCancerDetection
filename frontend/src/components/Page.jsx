function Page({children, className}) {
    return (
        <div className={`min-h-screen w-full bg-white ${className}`}>
            {children}
        </div>
    )
}

export default Page;