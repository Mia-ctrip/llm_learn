import { CameraView, useCameraPermissions } from 'expo-camera';
import { router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import {
  ActivityIndicator,
  Linking,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { AppButton } from '@/components/app-button';
import { AppScreen } from '@/components/app-screen';
import { BrandHeader } from '@/components/brand-header';
import { InlineNotice } from '@/components/inline-notice';
import { colors, spacing } from '@/constants/theme';
import { cameraPermissionState } from '@/lib/camera-permission';
import { CHECK_IN_VIEWS } from '@/lib/check-in-flow';
import { userFacingError } from '@/lib/errors';

export default function CheckInScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [focused, setFocused] = useState(false);
  const [requesting, setRequesting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const permissionState = cameraPermissionState(permission);
  const currentView = CHECK_IN_VIEWS[0];

  useFocusEffect(
    useCallback(() => {
      setFocused(true);
      return () => setFocused(false);
    }, []),
  );

  async function askForPermission() {
    setRequesting(true);
    setError(null);
    try {
      await requestPermission();
    } catch (requestError) {
      setError(userFacingError(requestError));
    } finally {
      setRequesting(false);
    }
  }

  async function openSettings() {
    setError(null);
    try {
      await Linking.openSettings();
    } catch (settingsError) {
      setError(userFacingError(settingsError));
    }
  }

  if (permissionState === 'loading') {
    return (
      <SafeAreaView style={styles.loading}>
        <ActivityIndicator color={colors.primary} size="large" />
        <Text style={styles.loadingText}>正在检查相机权限</Text>
      </SafeAreaView>
    );
  }

  if (permissionState !== 'granted') {
    const needsSettings = permissionState === 'settings';
    return (
      <AppScreen>
        <BrandHeader
          eyebrow="TODAY CHECK-IN"
          title={needsSettings ? '相机权限已关闭' : '允许使用相机'}
          description={
            needsSettings
              ? '请在系统设置中允许相机权限，然后返回继续今日 Check-in。'
              : '需要相机权限拍摄正面、左侧和右侧皮肤照片。'
          }
        />
        {error ? <InlineNotice tone="error" message={error} /> : null}
        <View style={styles.permissionActions}>
          <AppButton
            label={needsSettings ? '打开系统设置' : '允许使用相机'}
            loading={requesting}
            onPress={() =>
              void (needsSettings ? openSettings() : askForPermission())
            }
          />
          <AppButton label="返回首页" variant="text" onPress={() => router.back()} />
        </View>
      </AppScreen>
    );
  }

  return (
    <View style={styles.cameraScreen}>
      {focused ? (
        <CameraView
          facing="front"
          mirror={false}
          mode="picture"
          style={StyleSheet.absoluteFill}
        />
      ) : null}
      <SafeAreaView pointerEvents="box-none" style={styles.cameraOverlay}>
        <View style={styles.topBar}>
          <Pressable
            accessibilityLabel="返回首页"
            accessibilityRole="button"
            hitSlop={12}
            onPress={() => router.back()}
            style={({ pressed }) => [
              styles.backButton,
              pressed && styles.overlayPressed,
            ]}>
            <Text style={styles.backIcon}>‹</Text>
          </Pressable>
          <View style={styles.progressCopy}>
            <Text style={styles.progressLabel}>今日 Check-in</Text>
            <Text style={styles.progressValue}>1 / 3 · {currentView.label}</Text>
          </View>
          <View style={styles.topSpacer} />
        </View>

        <View style={styles.guideArea} pointerEvents="none">
          <View style={styles.faceGuide} />
        </View>

        <View style={styles.instruction}>
          <Text style={styles.instructionTitle}>{currentView.label}</Text>
          <Text style={styles.instructionText}>{currentView.instruction}</Text>
        </View>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  loading: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.lg,
    backgroundColor: colors.background,
  },
  loadingText: {
    color: colors.textMuted,
    fontSize: 14,
  },
  permissionActions: {
    gap: spacing.sm,
    marginTop: spacing.xxl,
  },
  cameraScreen: {
    flex: 1,
    backgroundColor: colors.text,
  },
  cameraOverlay: {
    flex: 1,
    justifyContent: 'space-between',
  },
  topBar: {
    minHeight: 64,
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    backgroundColor: 'rgba(20, 24, 21, 0.58)',
  },
  backButton: {
    width: 44,
    height: 44,
    alignItems: 'center',
    justifyContent: 'center',
  },
  backIcon: {
    color: colors.white,
    fontSize: 40,
    lineHeight: 42,
  },
  overlayPressed: {
    opacity: 0.7,
  },
  progressCopy: {
    flex: 1,
    alignItems: 'center',
    gap: 2,
  },
  progressLabel: {
    color: colors.white,
    fontSize: 15,
    fontWeight: '700',
  },
  progressValue: {
    color: '#E6ECE8',
    fontSize: 12,
  },
  topSpacer: {
    width: 44,
  },
  guideArea: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: spacing.xxl,
  },
  faceGuide: {
    width: '82%',
    maxWidth: 330,
    aspectRatio: 0.72,
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.9)',
    borderRadius: 999,
  },
  instruction: {
    gap: spacing.sm,
    paddingHorizontal: spacing.xl,
    paddingTop: spacing.lg,
    paddingBottom: spacing.xxl,
    backgroundColor: 'rgba(20, 24, 21, 0.72)',
  },
  instructionTitle: {
    color: colors.white,
    fontSize: 20,
    fontWeight: '800',
    textAlign: 'center',
  },
  instructionText: {
    color: '#F1F4F2',
    fontSize: 14,
    lineHeight: 21,
    textAlign: 'center',
  },
});
