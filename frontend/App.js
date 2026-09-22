import { useEffect, useState } from 'react';
import * as ImagePicker from 'expo-image-picker';
import { StatusBar } from 'expo-status-bar';
import {
  ActivityIndicator, Image, Pressable,
  ScrollView, StyleSheet, Text, View,
} from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';

const API_URL = 'https://carlens-qmnn.onrender.com';
const RETRY_DELAY_MS = 3000;
const SLOW_AFTER_MS = 4000;

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export default function App() {
  const [imageUri, setImageUri] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [slow, setSlow] = useState(false);

  // Render's free tier sleeps when idle. Wake it as soon as the app opens,
  // so it's usually ready by the time the user has picked a photo.
  useEffect(() => {
    fetch(`${API_URL}/health`).catch(() => {});
  }, []);

  // Picking an image

  async function pickFromLibrary() {
    const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!perm.granted) return setError('Photo library permission is required.');

    const picked = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      quality: 0.8,
    });
    if (!picked.canceled) analyze(picked.assets[0]);
  }

  async function takePhoto() {
    const perm = await ImagePicker.requestCameraPermissionsAsync();
    if (!perm.granted) return setError('Camera permission is required.');

    const picked = await ImagePicker.launchCameraAsync({
      mediaTypes: ['images'],
      quality: 0.8,
    });
    if (!picked.canceled) analyze(picked.assets[0]);
  }

  // Uploading to the API

  async function analyze(asset) {
    setImageUri(asset.uri);
    setResult(null);
    setError(null);
    setLoading(true);

    const name = asset.fileName ?? 'photo.jpg';
    const ext = name.split('.').pop().toLowerCase();
    const type =
      ext === 'png' ? 'image/png' : ext === 'webp' ? 'image/webp' : 'image/jpeg';

    const upload = () => {
      const form = new FormData();
      form.append('file', { uri: asset.uri, name, type });
      return fetch(`${API_URL}/predict`, { method: 'POST', body: form });
    };

    const slowTimer = setTimeout(() => setSlow(true), SLOW_AFTER_MS);

    try {
      let res;
      try {
        res = await upload();
      } catch {
        // No response at all, usually the server still waking. Retry once.
        await wait(RETRY_DELAY_MS);
        res = await upload();
      }

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setError(describe(res.status, body.detail));
        return;
      }
      setResult(await res.json());
    } catch {
      setError('Could not reach the server. Check your connection and try again.');
    } finally {
      clearTimeout(slowTimer);
      setSlow(false);
      setLoading(false);
    }
  }

  // Rendering

  return (
    <SafeAreaProvider>
      <SafeAreaView style={styles.safe}>
        <ScrollView contentContainerStyle={styles.container}>
          <Text style={styles.title}>CarLens</Text>
          <Text style={styles.subtitle}>
            Pick a photo and the model returns its five most likely guesses.
          </Text>
          <Text style={styles.note}>
            This is an ImageNet baseline, so it recognises broad categories like
            "sports car" rather than specific makes and models.
          </Text>

          <View style={styles.buttons}>
            <Pressable style={styles.button} onPress={pickFromLibrary}>
              <Text style={styles.buttonText}>Choose photo</Text>
            </Pressable>
            <Pressable style={[styles.button, styles.secondary]} onPress={takePhoto}>
              <Text style={[styles.buttonText, styles.secondaryText]}>Take photo</Text>
            </Pressable>
          </View>

          {imageUri && <Image source={{ uri: imageUri }} style={styles.preview} />}

          {loading && (
            <View style={styles.center}>
              <ActivityIndicator />
              <Text style={[styles.muted, styles.centerText]}>
                {slow
                  ? 'Waking up the server. The free tier sleeps when idle, so this can take up to a minute.'
                  : 'Analysing...'}
              </Text>
            </View>
          )}

          {error && <Text style={styles.error}>{error}</Text>}

          {result?.predictions.map((p, i) => (
            <View key={p.label} style={styles.row}>
              <View style={styles.rowHead}>
                <Text style={[styles.label, i === 0 && styles.topLabel]}>{p.label}</Text>
                <Text style={styles.pct}>{(p.confidence * 100).toFixed(1)}%</Text>
              </View>
              <View style={styles.barBg}>
                <View
                  style={[
                    styles.barFill,
                    { width: `${Math.max(p.confidence * 100, 1)}%` },
                    i === 0 && styles.topBar,
                  ]}
                />
              </View>
            </View>
          ))}

          {result && (
            <Text style={styles.muted}>Inference took {result.latency_ms} ms.</Text>
          )}
        </ScrollView>
        <StatusBar style="auto" />
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

function describe(status, detail) {
  if (status === 415) return 'Unsupported file type. Use a JPEG, PNG or WebP image.';
  if (status === 413) return 'That image is over the 8 MB limit.';
  if (status === 400) return 'That file could not be read as an image.';
  return detail ?? 'Something went wrong. Please try again.';
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#fbfbfd' },
  container: { padding: 24, paddingBottom: 48 },
  title: { fontSize: 32, fontWeight: '700', marginBottom: 6 },
  subtitle: { fontSize: 16, color: '#6b6b76', marginBottom: 16 },
  note: {
    fontSize: 14, backgroundColor: '#fff8e6', borderColor: '#f0dda6',
    borderWidth: 1, borderRadius: 12, padding: 12, marginBottom: 20,
  },
  buttons: { flexDirection: 'row', gap: 12, marginBottom: 20 },
  button: {
    flex: 1, backgroundColor: '#2f6fed', paddingVertical: 14,
    borderRadius: 12, alignItems: 'center',
  },
  buttonText: { color: '#fff', fontWeight: '600', fontSize: 16 },
  secondary: { backgroundColor: '#e8effd' },
  secondaryText: { color: '#2f6fed' },
  preview: {
    width: '100%', height: 260, borderRadius: 12, marginBottom: 20,
    backgroundColor: '#eeeef2', resizeMode: 'contain',
  },
  center: { alignItems: 'center', gap: 8, marginVertical: 12 },
  centerText: { textAlign: 'center' },
  muted: { color: '#6b6b76', fontSize: 14, marginTop: 8 },
  error: { color: '#c0392b', fontSize: 15, marginVertical: 12 },
  row: { marginBottom: 14 },
  rowHead: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 5 },
  label: { fontSize: 16, fontWeight: '500' },
  topLabel: { color: '#2f6fed', fontWeight: '700' },
  pct: { color: '#6b6b76', fontVariant: ['tabular-nums'] },
  barBg: { height: 7, backgroundColor: '#eeeef2', borderRadius: 4, overflow: 'hidden' },
  barFill: { height: '100%', backgroundColor: '#2f6fed', opacity: 0.45, borderRadius: 4 },
  topBar: { opacity: 1 },
});
