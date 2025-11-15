export default function Pagination ( {currentPage, setCurrentPage, totalItems, rowsPerPage} ) {

    const totalPages = Math.ceil(totalItems / rowsPerPage)
    console.log("--------------------")
    console.log(currentPage)
    console.log(totalPages)
    if (totalPages <= 1) return null;
    return (
        <div className="flex items-center justify-center gap-4 mt-4">

            <button
                disabled={currentPage == 1}
                onClick={() => setCurrentPage((prev) => prev - 1)}
                className={`px-3 py-1 rounded border border-gray-700 
                    ${currentPage === 1 ? "text-gray-600" : "hover:bg-gray-800"}`}
            >
                Previous
            </button>
            
            {/* page numbers */ }
            <div className="flex gap-2">
                {Array.from({length: totalPages}, (_,i)=>i+1).map((num) => (
                  <button
                    key={num}
                    onClick={() => setCurrentPage(num)}
                    className={`px-3 py-1 rounded ${
                                    num === currentPage
                                        ? "bg-blue-600 text-white"
                                        : "bg-gray-800 hover:bg-gray-700"
                                }`
                            }
                  >
                    {num}
                  </button>  
                ))}
            </div>

            {/* Next */}
            <button
                disabled={currentPage == totalPages}
                onClick={() => setCurrentPage((prev) => prev + 1)}
                className={`px-3 py-1 rounded border border-gray-700 
                    ${currentPage === totalPages ? "text-gray-600" : "hover:bg-gray-800"}`}
            >
                Next
            </button>
        </div>
    );
}