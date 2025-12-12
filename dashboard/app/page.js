import UploadVideo from "../components/UploadVideo";

export default function Home() {
  return (
    <div>
      <h1 className="text-2xl font-bold mb-6 text-black">Upload Video</h1>

      <div className="bg-white shadow p-6 rounded-md w-fit">
        <UploadVideo />
      </div>
    </div>
  );
}
