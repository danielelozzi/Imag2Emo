<?php
// Set the content type to plain text to ensure the browser interprets the CSV correctly.
header('Content-Type: text/plain');
// Allow requests from any origin. This is important for testing.
// For production, you might want to restrict this to your specific domain.
header('Access-Control-Allow-Origin: *');

// Check if the 'path' parameter is provided in the URL.
if (isset($_GET['path'])) {
    // The path from JS will be like "DEAP/PUBLIC/..."
    // We need to prepend the main results directory to it.
    $filePath = 'results/' . $_GET['path'];

    // --- Basic Security Check ---
    // 1. Use realpath to resolve any '..' or '.' in the path.
    // 2. Ensure the resolved path is within your allowed 'results' directory.
    $baseDir = realpath(__DIR__ . '/results');
    $requestedPath = realpath(__DIR__ . '/' . $filePath);

    // Check if the requested path starts with the base directory path.
    // This prevents directory traversal attacks (e.g., accessing files outside 'results').
    if ($requestedPath && strpos($requestedPath, $baseDir) === 0 && file_exists($requestedPath)) {
        // If the file exists and is within the allowed directory, read and output its content.
        echo file_get_contents($requestedPath);
    } else {
        // If the file doesn't exist or the path is invalid/unsafe, return a 404 error.
        http_response_code(404);
        echo "Error: File not found or access denied at '{$filePath}'";
    }
} else {
    // If the 'path' parameter is missing, return a 400 Bad Request error.
    http_response_code(400);
    echo "Error: 'path' parameter is missing.";
}
?>
