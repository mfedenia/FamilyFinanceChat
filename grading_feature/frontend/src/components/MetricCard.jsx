
export default function MetricCard ( {title, value} ) {
    return (
        <div className="bg-[#1a1a1a] border border-gray-800 rounded-xl p-4 shadow-sm">
        <p className="text-gray-400 text-sm">{title}</p>
        <p className="text-3xl font-semibold text-gray-100 mt-2">{value}</p>
        </div>
    );

}